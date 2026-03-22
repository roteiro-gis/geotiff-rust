#!/usr/bin/env python3

import argparse
import json
import statistics
import sys
import time

from osgeo import gdal
from osgeo import osr


def fnv1a64(data: bytes) -> str:
    hash_value = 0xCBF29CE484222325
    for byte in data:
        hash_value ^= byte
        hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{hash_value:016x}"


def dataset_or_exit(path: str):
    gdal.UseExceptions()
    dataset = gdal.Open(path, gdal.GA_ReadOnly)
    if dataset is None:
        raise SystemExit(f"failed to open dataset: {path}")
    return dataset


def band_bytes(band) -> bytes:
    width = band.XSize
    height = band.YSize
    data_type = band.DataType
    data = band.ReadRaster(0, 0, width, height, buf_type=data_type)
    if data is None:
        raise SystemExit("failed to read raster bytes from GDAL")
    return data


def interleaved_bytes(dataset, overview_index):
    bands = []
    for band_index in range(1, dataset.RasterCount + 1):
        band = dataset.GetRasterBand(band_index)
        if overview_index is not None:
            band = band.GetOverview(overview_index)
            if band is None:
                raise SystemExit(f"overview {overview_index} not found for band {band_index}")
        bands.append(band)

    width = bands[0].XSize
    height = bands[0].YSize
    data_type = bands[0].DataType
    sample_size = gdal.GetDataTypeSize(data_type) // 8
    pixel_count = width * height
    per_band = [band_bytes(band) for band in bands]

    if len(per_band) == 1:
        return per_band[0], width, height, data_type

    out = bytearray(pixel_count * len(per_band) * sample_size)
    for band_offset, payload in enumerate(per_band):
        for pixel_index in range(pixel_count):
            src_start = pixel_index * sample_size
            dst_start = (pixel_index * len(per_band) + band_offset) * sample_size
            out[dst_start : dst_start + sample_size] = payload[src_start : src_start + sample_size]
    return bytes(out), width, height, data_type


def extract_epsg(dataset):
    projection = dataset.GetProjectionRef()
    if not projection:
        return None
    spatial_ref = osr.SpatialReference()
    spatial_ref.ImportFromWkt(projection)
    try:
        spatial_ref.AutoIdentifyEPSG()
    except RuntimeError:
        pass
    for authority_target in ("PROJCS", "GEOGCS", None):
        code = spatial_ref.GetAuthorityCode(authority_target)
        if code:
            return int(code)
    return None


def command_metadata(args):
    dataset = dataset_or_exit(args.path)
    geo_transform = dataset.GetGeoTransform(can_return_null=True)
    payload = {
        "width": dataset.RasterXSize,
        "height": dataset.RasterYSize,
        "band_count": dataset.RasterCount,
        "overview_count": dataset.GetRasterBand(1).GetOverviewCount() if dataset.RasterCount else 0,
        "compression": dataset.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE"),
        "interleave": dataset.GetMetadataItem("INTERLEAVE", "IMAGE_STRUCTURE"),
        "area_or_point": dataset.GetMetadataItem("AREA_OR_POINT"),
        "geo_transform": list(geo_transform) if geo_transform is not None else None,
        "epsg": extract_epsg(dataset),
        "band_data_types": [
            gdal.GetDataTypeName(dataset.GetRasterBand(index).DataType)
            for index in range(1, dataset.RasterCount + 1)
        ],
        "color_interpretations": [
            gdal.GetColorInterpretationName(dataset.GetRasterBand(index).GetColorInterpretation())
            for index in range(1, dataset.RasterCount + 1)
        ],
    }
    print(json.dumps(payload))


def command_hash(args):
    dataset = dataset_or_exit(args.path)
    data, width, height, data_type = interleaved_bytes(dataset, args.overview)
    payload = {
        "width": width,
        "height": height,
        "band_count": dataset.RasterCount,
        "data_type": gdal.GetDataTypeName(data_type),
        "byte_len": len(data),
        "hash": fnv1a64(data),
    }
    print(json.dumps(payload))


def command_bytes(args):
    dataset = dataset_or_exit(args.path)
    data, _, _, _ = interleaved_bytes(dataset, args.overview)
    sys.stdout.buffer.write(data)


def command_benchmark(args):
    timings = []
    hashes = set()
    byte_len = None

    for _ in range(args.iterations):
        start = time.perf_counter()
        dataset = dataset_or_exit(args.path)
        data, _, _, _ = interleaved_bytes(dataset, None)
        timings.append(time.perf_counter() - start)
        hashes.add(fnv1a64(data))
        byte_len = len(data)

    if len(hashes) != 1:
        raise SystemExit("GDAL benchmark did not produce a stable raster hash")

    payload = {
        "iterations": args.iterations,
        "min_seconds": min(timings),
        "median_seconds": statistics.median(timings),
        "max_seconds": max(timings),
        "byte_len": byte_len,
        "hash": next(iter(hashes)),
    }
    print(json.dumps(payload))


def build_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser("metadata")
    metadata.add_argument("path")
    metadata.set_defaults(func=command_metadata)

    hash_parser = subparsers.add_parser("hash")
    hash_parser.add_argument("path")
    hash_parser.add_argument("--overview", type=int, default=None)
    hash_parser.set_defaults(func=command_hash)

    bytes_parser = subparsers.add_parser("bytes")
    bytes_parser.add_argument("path")
    bytes_parser.add_argument("--overview", type=int, default=None)
    bytes_parser.set_defaults(func=command_bytes)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("path")
    benchmark.add_argument("--iterations", type=int, default=5)
    benchmark.set_defaults(func=command_benchmark)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as err:
        print(str(err), file=sys.stderr)
        raise
