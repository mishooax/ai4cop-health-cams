# Utility functions for downloading data from CAMS ADS / CDS


"""
import cdsapi

c = cdsapi.Client()

c.retrieve(
    'cams-europe-air-quality-reanalyses',
    {
        'year': '2018',
        'format': 'zip',
        'variable': [
            'ammonia', 'carbon_monoxide', 'dust',
            'nitrogen_dioxide', 'nitrogen_monoxide', 'non_methane_vocs',
            'ozone', 'particulate_matter_10um', 'particulate_matter_2.5um',
            'sulphur_dioxide',
        ],
        'model': 'ensemble',
        'level': '0_m',
        'type': 'validated_reanalysis',
        'month': '01',
    },
    'download.zip')
"""

import datetime as dt
import calendar
import argparse
import os
import cdsapi

from ai4cop_health_cams.config import YAMLConfig
from ai4cop_health_cams.logger import get_logger


LOGGER = get_logger(__name__)


def download_eu_reanalysis_data(config: YAMLConfig) -> None:
    c = cdsapi.Client()
    for month in range(3, 13):
        download_fpath = os.path.join(
            config["input:raw:basedir"],
            f"eu/analysis/zip/download_2018_{month:02d}.zip",
        )
        os.makedirs(os.path.dirname(download_fpath), exist_ok=True)
        c.retrieve(
            "cams-europe-air-quality-reanalyses",
            {
                "year": "2018",
                "format": "zip",
                "variable": [
                    "ammonia",
                    "carbon_monoxide",
                    "dust",
                    "nitrogen_dioxide",
                    "nitrogen_monoxide",
                    "non_methane_vocs",
                    "ozone",
                    "particulate_matter_10um",
                    "particulate_matter_2.5um",
                    "sulphur_dioxide",
                ],
                "model": "ensemble",
                "level": "0_m",
                "type": "validated_reanalysis",
                "month": f"{month:02d}",
            },
            download_fpath,
        )


def download_eu_forecast_data(start_year: int, end_year: int) -> None:
    """Download European (hi-res) AQ data."""
    c = cdsapi.Client()

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            _, n_days = calendar.monthrange(year, month)
            date_start = dt.datetime(year, month, 1, 0)
            date_end = dt.datetime(year, month, n_days, 0)
            c.retrieve(
                "cams-europe-air-quality-forecasts",
                {
                    # air quality variables
                    "variable": [
                        "ammonia",
                        "carbon_monoxide",
                        "nitrogen_dioxide",
                        "nitrogen_monoxide",
                        "ozone",
                        "particulate_matter_10um",
                        "particulate_matter_2.5um",
                        "sulphur_dioxide",
                    ],
                    "model": "ensemble",  # ensemble median
                    "level": "0",  # height above surface [m]
                    "date": "{d_start}/{d_end}".format(
                        d_start=date_start.strftime("%Y-%m-%d"),
                        d_end=date_end.strftime("%Y-%m-%d"),
                    ),
                    "type": "analysis",
                    # 3-hourly time steps (same as GLB reanalysis)
                    "time": [
                        "00:00",
                        "03:00",
                        "06:00",
                        "09:00",
                        "12:00",
                        "15:00",
                        "18:00",
                        "21:00",
                    ],
                    "area": [
                        # N, W, S, E limits
                        70.0,
                        -25.0,
                        30.0,
                        45.0,
                    ],
                    "leadtime_hour": "0",
                    "format": "netcdf",
                },
                # single netCDF file output
                "../eu/cams_eu_aq_{d_start}_{d_end}.nc".format(
                    d_start=date_start.strftime("%Y%m%d%H"),
                    d_end=date_end.strftime("%Y%m%d%H"),
                ),
            )


def download_global_data(start_date: str, end_date: str, config: YAMLConfig) -> None:
    c = cdsapi.Client()
    sdate = dt.datetime.strptime(start_date, "%Y%m%d")
    edate = dt.datetime.strptime(end_date, "%Y%m%d")

    for year in range(sdate.year, edate.year + 1):
        for month in range(1, 13):
            _, n_days = calendar.monthrange(year, month)
            date_start = dt.datetime(year, month, 1, 0)
            date_end = dt.datetime(year, month, n_days, 0)

            c.retrieve(
                "cams-global-reanalysis-eac4",
                {
                    "variable": [
                        "10m_u_component_of_wind",
                        "10m_v_component_of_wind",
                        "2m_temperature",
                        "ammonia",
                        "carbon_monoxide",
                        "land_sea_mask",
                        "nitrogen_dioxide",
                        "nitrogen_monoxide",
                        "ozone",
                        "particulate_matter_10um",
                        "particulate_matter_2.5um",
                        "peroxyacetyl_nitrate",
                        "sulphur_dioxide",
                        "surface_geopotential",
                        "surface_pressure",
                        "u_component_of_wind",
                        "v_component_of_wind",
                    ],
                    "model_level": "60",
                    "date": "{d_start}/{d_end}".format(
                        d_start=date_start.strftime("%Y-%m-%d"),
                        d_end=date_end.strftime("%Y-%m-%d"),
                    ),
                    "time": [
                        "00:00",
                        "03:00",
                        "06:00",
                        "09:00",
                        "12:00",
                        "15:00",
                        "18:00",
                        "21:00",
                    ],
                    "area": [
                        70.0,
                        -25.0,
                        30.0,
                        45.0,
                    ],
                    "format": "netcdf",
                },
                # single zip file output:
                # contains two files levtype_ml.nc (model-level vars) and levtype_sfc.nc (surface vars)
                # unpack with:
                #   mkdir tmp
                #   for f in *.zip; do unzip "$f" -d tmp && mv tmp/*_ml.nc "${f%.zip}_ml.nc" && mv tmp/*_sfc.nc "${f%.zip}_sfc.nc"; done
                #   rmdir tmp
                os.path.join(
                    config["input:raw:basedir"],
                    "glb/reanalysis/cams_glb_eac4_{d_start}_{d_end}.zip".format(
                        d_start=date_start.strftime("%Y%m%d%H"),
                        d_end=date_end.strftime("%Y%m%d%H"),
                    ),
                ),
            )


def get_args() -> argparse.Namespace:
    """Returns a namespace containing the command line arguments"""
    parser = argparse.ArgumentParser()
    required_args = parser.add_argument_group("required arguments")
    required_args.add_argument("--start-date", required=True, help="Start date [YYYYMMDD]")
    required_args.add_argument("--end-date", required=True, help="Start date [YYYYMMDD]")
    required_args.add_argument(
        "--dataset",
        required=True,
        choices=["eu_analysis", "eu_reanalysis", "glb_reanalysis", "eu_ensemble"],
        help="Dataset to download (Europe or global)",
    )
    required_args.add_argument("--config", required=True, help="Model configuration file (YAML)")
    return parser.parse_args()


def download_eu_ensemble_data(config: YAMLConfig) -> None:

    c = cdsapi.Client()
    models = ["chimere", "dehm", "emep", "euradim", "gemaq", "lotos", "match", "mocage", "silam"]
    for model in models:
        download_fpath = os.path.join(
            config["input:raw:basedir"],
            f"eu/analysis/cams_eu_aq_ens_{model}_2019070100_2019123100.nc",
        )
        os.makedirs(os.path.dirname(download_fpath), exist_ok=True)
        c.retrieve(
            "cams-europe-air-quality-forecasts",
            {
                "variable": "particulate_matter_2.5um",
                "model": model,
                "level": "0",
                "date": "2019-07-01/2019-12-31",
                "type": "analysis",
                "time": [
                    "00:00",
                    "01:00",
                    "02:00",
                    "03:00",
                    "04:00",
                    "05:00",
                    "06:00",
                    "07:00",
                    "08:00",
                    "09:00",
                    "10:00",
                    "11:00",
                    "12:00",
                    "13:00",
                    "14:00",
                    "15:00",
                    "16:00",
                    "17:00",
                    "18:00",
                    "19:00",
                    "20:00",
                    "21:00",
                    "22:00",
                    "23:00",
                ],
                "leadtime_hour": "0",
                "area": [
                    60.32,
                    -12.95,
                    34.87,
                    38.58,
                ],
                "format": "netcdf",
            },
            download_fpath,
        )


def main():
    """Data downloader ..."""
    args = get_args()
    config = YAMLConfig(args.config)

    if args.dataset == "glb_reanalysis":
        download_global_data(args.start_date, args.end_date, config)
    elif args.dataset == "eu_analysis":
        download_eu_forecast_data(args.start_date, args.end_date)
    elif args.dataset == "eu_reanalysis":
        download_eu_reanalysis_data(config)
    elif args.dataset == "eu_ensemble":
        download_eu_ensemble_data(config)
    else:
        raise RuntimeError("Invalid dataset name.")

    LOGGER.debug("---- DONE. ----")
