//! Low-level TIFF byte emission: header writing, IFD serialization, offset patching.

use std::io::{Seek, SeekFrom, Write};

use tiff_core::{ByteOrder, Tag, TagValue};

use crate::error::Result;

/// Write the TIFF header. Classic = 8 bytes, BigTIFF = 16 bytes.
/// The first-IFD offset is set to 0 and must be patched later.
pub fn write_header<W: Write + Seek>(
    sink: &mut W,
    byte_order: ByteOrder,
    is_bigtiff: bool,
) -> Result<u64> {
    let pos = sink.stream_position()?;
    sink.write_all(&byte_order.magic())?;
    if is_bigtiff {
        sink.write_all(&byte_order.write_u16(43))?;
        sink.write_all(&byte_order.write_u16(8))?; // offset size
        sink.write_all(&byte_order.write_u16(0))?; // reserved
        sink.write_all(&byte_order.write_u64(0))?; // placeholder
    } else {
        sink.write_all(&byte_order.write_u16(42))?;
        sink.write_all(&byte_order.write_u32(0))?; // placeholder
    }
    Ok(pos)
}

/// Patch the first-IFD offset in the file header.
pub fn patch_first_ifd<W: Write + Seek>(
    sink: &mut W,
    header_offset: u64,
    byte_order: ByteOrder,
    is_bigtiff: bool,
    ifd_offset: u64,
) -> Result<()> {
    if is_bigtiff {
        sink.seek(SeekFrom::Start(header_offset + 8))?;
        sink.write_all(&byte_order.write_u64(ifd_offset))?;
    } else {
        sink.seek(SeekFrom::Start(header_offset + 4))?;
        sink.write_all(&byte_order.write_u32(ifd_offset as u32))?;
    }
    Ok(())
}

/// State returned after writing an IFD, used for patching.
pub struct IfdWriteResult {
    /// File offset where this IFD starts.
    pub ifd_offset: u64,
    /// File offset of the "next IFD" pointer.
    pub next_ifd_pointer_offset: u64,
    /// File offsets where the offset-array and bytecount-array deferred data reside.
    pub offsets_tag_data_offset: Option<u64>,
    pub byte_counts_tag_data_offset: Option<u64>,
    /// Whether this IFD was written in BigTIFF format.
    pub is_bigtiff: bool,
}

/// Write an IFD (Classic or BigTIFF). Tags must be sorted by code.
pub fn write_ifd<W: Write + Seek>(
    sink: &mut W,
    byte_order: ByteOrder,
    is_bigtiff: bool,
    tags: &[Tag],
    offsets_tag_code: u16,
    byte_counts_tag_code: u16,
    _num_blocks: usize,
) -> Result<IfdWriteResult> {
    let ifd_offset = sink.stream_position()?;

    // Sizes depend on format
    let entry_size: u64 = if is_bigtiff { 20 } else { 12 };
    let inline_max: usize = if is_bigtiff { 8 } else { 4 };
    let next_ptr_size: u64 = if is_bigtiff { 8 } else { 4 };
    let count_size: u64 = if is_bigtiff { 8 } else { 2 };

    // Entry count
    if is_bigtiff {
        sink.write_all(&byte_order.write_u64(tags.len() as u64))?;
    } else {
        sink.write_all(&byte_order.write_u16(tags.len() as u16))?;
    }

    // Calculate deferred data area start
    let entries_total = tags.len() as u64 * entry_size;
    let deferred_start = ifd_offset + count_size + entries_total + next_ptr_size;
    let mut deferred_offset = deferred_start;

    struct DeferredEntry {
        data: Vec<u8>,
        offset: u64,
    }
    let mut deferred_entries = Vec::new();
    let mut offsets_data_offset = None;
    let mut byte_counts_data_offset = None;

    // First pass: determine which tags are deferred and their offsets
    for tag in tags {
        let encoded = tag.value.encode(byte_order);
        if encoded.len() > inline_max {
            if tag.code == offsets_tag_code {
                offsets_data_offset = Some(deferred_offset);
            } else if tag.code == byte_counts_tag_code {
                byte_counts_data_offset = Some(deferred_offset);
            }
            let len = encoded.len() as u64;
            deferred_entries.push(DeferredEntry {
                data: encoded,
                offset: deferred_offset,
            });
            deferred_offset += len;
        }
    }

    // Second pass: write entries
    let mut deferred_idx = 0;
    for tag in tags {
        sink.write_all(&byte_order.write_u16(tag.code))?;
        sink.write_all(&byte_order.write_u16(tag.tag_type.to_code()))?;

        if is_bigtiff {
            sink.write_all(&byte_order.write_u64(tag.count))?;
        } else {
            sink.write_all(&byte_order.write_u32(tag.count as u32))?;
        }

        let encoded = tag.value.encode(byte_order);
        if encoded.len() <= inline_max {
            let mut inline = vec![0u8; inline_max];
            inline[..encoded.len()].copy_from_slice(&encoded);
            sink.write_all(&inline)?;
        } else {
            let offset = deferred_entries[deferred_idx].offset;
            if is_bigtiff {
                sink.write_all(&byte_order.write_u64(offset))?;
            } else {
                sink.write_all(&byte_order.write_u32(offset as u32))?;
            }
            deferred_idx += 1;
        }
    }

    // Next-IFD pointer
    let next_ifd_pointer_offset = sink.stream_position()?;
    if is_bigtiff {
        sink.write_all(&byte_order.write_u64(0))?;
    } else {
        sink.write_all(&byte_order.write_u32(0))?;
    }

    // Write deferred data
    for entry in &deferred_entries {
        debug_assert_eq!(sink.stream_position()?, entry.offset);
        sink.write_all(&entry.data)?;
    }

    Ok(IfdWriteResult {
        ifd_offset,
        next_ifd_pointer_offset,
        offsets_tag_data_offset: offsets_data_offset,
        byte_counts_tag_data_offset: byte_counts_data_offset,
        is_bigtiff,
    })
}

/// Patch the block offsets array in a previously written IFD.
pub fn patch_block_offsets<W: Write + Seek>(
    sink: &mut W,
    byte_order: ByteOrder,
    is_bigtiff: bool,
    data_offset: u64,
    offsets: &[u64],
) -> Result<()> {
    sink.seek(SeekFrom::Start(data_offset))?;
    for &offset in offsets {
        if is_bigtiff {
            sink.write_all(&byte_order.write_u64(offset))?;
        } else {
            sink.write_all(&byte_order.write_u32(offset as u32))?;
        }
    }
    Ok(())
}

/// Patch the block byte-counts array in a previously written IFD.
pub fn patch_block_byte_counts<W: Write + Seek>(
    sink: &mut W,
    byte_order: ByteOrder,
    is_bigtiff: bool,
    data_offset: u64,
    byte_counts: &[u64],
) -> Result<()> {
    sink.seek(SeekFrom::Start(data_offset))?;
    for &count in byte_counts {
        if is_bigtiff {
            sink.write_all(&byte_order.write_u64(count))?;
        } else {
            sink.write_all(&byte_order.write_u32(count as u32))?;
        }
    }
    Ok(())
}

/// Patch the next-IFD pointer.
pub fn patch_next_ifd<W: Write + Seek>(
    sink: &mut W,
    byte_order: ByteOrder,
    is_bigtiff: bool,
    pointer_offset: u64,
    next_ifd: u64,
) -> Result<()> {
    sink.seek(SeekFrom::Start(pointer_offset))?;
    if is_bigtiff {
        sink.write_all(&byte_order.write_u64(next_ifd))?;
    } else {
        sink.write_all(&byte_order.write_u32(next_ifd as u32))?;
    }
    Ok(())
}

/// Parameters for building image tags.
pub struct ImageTagParams<'a> {
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: u16,
    pub bits_per_sample: u16,
    pub sample_format: u16,
    pub compression: u16,
    pub photometric: u16,
    pub predictor: u16,
    pub planar_configuration: u16,
    pub subfile_type: u32,
    pub extra_tags: &'a [Tag],
    pub offsets_tag_code: u16,
    pub byte_counts_tag_code: u16,
    pub num_blocks: usize,
    pub layout_tags: &'a [Tag],
    pub is_bigtiff: bool,
}

/// Build standard TIFF tags for an image.
/// For BigTIFF, offset/bytecount arrays use Long8 instead of Long.
pub fn build_image_tags(p: &ImageTagParams<'_>) -> Vec<Tag> {
    let ImageTagParams {
        width,
        height,
        samples_per_pixel,
        bits_per_sample,
        sample_format,
        compression,
        photometric,
        predictor,
        planar_configuration,
        subfile_type,
        extra_tags,
        offsets_tag_code,
        byte_counts_tag_code,
        num_blocks,
        layout_tags,
        is_bigtiff,
    } = p;
    let mut tags = Vec::with_capacity(16 + extra_tags.len());

    if *subfile_type != 0 {
        tags.push(Tag::new(
            tiff_core::TAG_NEW_SUBFILE_TYPE,
            TagValue::Long(vec![*subfile_type]),
        ));
    }
    tags.push(Tag::new(
        tiff_core::TAG_IMAGE_WIDTH,
        TagValue::Long(vec![*width]),
    ));
    tags.push(Tag::new(
        tiff_core::TAG_IMAGE_LENGTH,
        TagValue::Long(vec![*height]),
    ));
    tags.push(Tag::new(
        tiff_core::TAG_BITS_PER_SAMPLE,
        TagValue::Short(vec![*bits_per_sample; *samples_per_pixel as usize]),
    ));
    tags.push(Tag::new(
        tiff_core::TAG_COMPRESSION,
        TagValue::Short(vec![*compression]),
    ));
    tags.push(Tag::new(
        tiff_core::TAG_PHOTOMETRIC_INTERPRETATION,
        TagValue::Short(vec![*photometric]),
    ));
    tags.push(Tag::new(
        tiff_core::TAG_SAMPLES_PER_PIXEL,
        TagValue::Short(vec![*samples_per_pixel]),
    ));
    if *planar_configuration != 1 {
        tags.push(Tag::new(
            tiff_core::TAG_PLANAR_CONFIGURATION,
            TagValue::Short(vec![*planar_configuration]),
        ));
    }
    if *predictor != 1 {
        tags.push(Tag::new(
            tiff_core::TAG_PREDICTOR,
            TagValue::Short(vec![*predictor]),
        ));
    }
    tags.push(Tag::new(
        tiff_core::TAG_SAMPLE_FORMAT,
        TagValue::Short(vec![*sample_format; *samples_per_pixel as usize]),
    ));

    for lt in *layout_tags {
        tags.push(lt.clone());
    }

    // Offset and bytecount placeholder arrays
    if *is_bigtiff {
        tags.push(Tag::new(
            *offsets_tag_code,
            TagValue::Long8(vec![0u64; *num_blocks]),
        ));
        tags.push(Tag::new(
            *byte_counts_tag_code,
            TagValue::Long8(vec![0u64; *num_blocks]),
        ));
    } else {
        tags.push(Tag::new(
            *offsets_tag_code,
            TagValue::Long(vec![0u32; *num_blocks]),
        ));
        tags.push(Tag::new(
            *byte_counts_tag_code,
            TagValue::Long(vec![0u32; *num_blocks]),
        ));
    }

    for tag in *extra_tags {
        tags.push(tag.clone());
    }

    tags.sort_by_key(|t| t.code);
    tags
}

/// Find the position of a tag's inline value within a written IFD.
pub fn find_inline_tag_value_offset(
    ifd_offset: u64,
    is_bigtiff: bool,
    tags: &[Tag],
    target_code: u16,
) -> Option<u64> {
    let count_size: u64 = if is_bigtiff { 8 } else { 2 };
    let entry_size: u64 = if is_bigtiff { 20 } else { 12 };
    // Value/offset field is the last 4 (classic) or 8 (BigTIFF) bytes of each entry.
    // Entry: code(2) + type(2) + count(4 or 8) + value(4 or 8)
    let value_field_offset: u64 = if is_bigtiff { 12 } else { 8 };

    for (i, tag) in tags.iter().enumerate() {
        if tag.code == target_code {
            return Some(ifd_offset + count_size + (i as u64) * entry_size + value_field_offset);
        }
    }
    None
}
