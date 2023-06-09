# Copyright 2023 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

""" Builders for generating Intake-ESM catalogs """

import multiprocessing
import re
import traceback
from pathlib import Path

import xarray as xr
from ecgtools.builder import INVALID_ASSET, TRACEBACK, Builder

from ..utils import validate_against_schema
from . import ESM_JSONSCHEMA, PATH_COLUMN, VARIABLE_COLUMN
from .utils import get_timeinfo, strip_pattern_rh


class ParserError(Exception):
    pass


class BaseBuilder(Builder):
    """
    Base class for creating intake-esm catalog builders. Not intended for direct use.
    This builds on the ecgtools.Builder class.
    """

    def __init__(
        self,
        path,
        depth=0,
        exclude_patterns=None,
        include_patterns=None,
        data_format="netcdf",
        groupby_attrs=None,
        aggregations=None,
        storage_options=None,
        joblib_parallel_kwargs={"n_jobs": multiprocessing.cpu_count()},
    ):
        """
        This method should be overwritten. The expection is that some of these arguments
        will be hardcoded in sub classes of this class.

        Parameters
        ----------
        path: str or list of str
            Path or list of path to crawl for assets/files.
        depth: int, optional
            Maximum depth to crawl for assets. Default is 0.
        exclude_patterns: list of str, optional
            List of glob patterns to exclude from crawling.
        include_patterns: list of str, optional
            List of glob patterns to include from crawling.
        data_format: str
            The data format. Valid values are 'netcdf', 'reference' and 'zarr'.
        groupby_attrs: List[str]
            Column names (attributes) that define data sets that can be aggegrated.
        aggregations: List[dict]
            List of aggregations to apply to query results, default None
        storage_options: dict, optional
            Parameters passed to the backend file-system such as Google Cloud Storage,
            Amazon Web Service S3
        joblib_parallel_kwargs: dict, optional
            Parameters passed to joblib.Parallel. Default is {}.
        """

        if isinstance(path, str):
            path = [path]

        self.paths = path
        self.depth = depth
        self.exclude_patterns = exclude_patterns
        self.include_patterns = include_patterns
        self.data_format = data_format
        self.groupby_attrs = groupby_attrs
        self.aggregations = aggregations
        self.storage_options = storage_options
        self.joblib_parallel_kwargs = joblib_parallel_kwargs

        super().__post_init_post_parse__()

    def _parse(self):
        super().parse(parsing_func=self.parser)

    def parse(self):
        """
        Parse metadata from assets.
        """
        self._parse()
        return self

    def _save(self, name, description, directory):
        super().save(
            name=name,
            path_column_name=PATH_COLUMN,
            variable_column_name=VARIABLE_COLUMN,
            data_format=self.data_format,
            groupby_attrs=self.groupby_attrs,
            aggregations=self.aggregations,
            esmcat_version="0.0.1",
            description=description,
            directory=directory,
            catalog_type="file",
            to_csv_kwargs={"compression": "gzip"},
        )

    def save(self, name, description, directory=None):
        """
        Save catalog contents to a file.

        Parameters
        ----------
        name: str
            The name of the file to save the catalog to.
        description : str
            Detailed multi-line description of the collection.
        directory: str, optional
            The directory to save the catalog to. If None, use the current directory.
        """

        if self.df.empty:
            raise ValueError(
                "intake-esm catalog has not yet been built. Please run `.build()` first"
            )

        self._save(name, description, directory)

    def validate_parser(self):
        """
        Run the parser on a single file and check the schema of the info being parsed
        """

        if not self.assets:
            raise ValueError(
                "asset list provided is None. Please run `.get_assets()` first"
            )

        for asset in self.assets:
            info = self.parser(asset)
            if INVALID_ASSET not in info:
                validate_against_schema(info, ESM_JSONSCHEMA)
                return self

        raise ParserError(
            "Parser returns no valid assets. Try parsing a single file with Builder.parser(file)"
        )

    def build(self):
        """
        Builds a catalog from a list of netCDF files or zarr stores.
        """

        self.get_assets().validate_parser().parse().clean_dataframe()

        return self

    @property
    def columns_with_iterables(self):
        """
        Return a set of the columns that have iterables
        """
        # Stolen from intake-esm.cat.ESMCatalogModel
        if self.df.empty:
            return set()
        has_iterables = (
            self.df.sample(20, replace=True)
            .applymap(type)
            .isin([list, tuple, set])
            .any()
            .to_dict()
        )
        return {column for column, check in has_iterables.items() if check}

    @staticmethod
    def parser(file):
        """
        Parse info from a file asset

        Parameters
        ----------
        file: str
            The path to the file
        """
        # This method should be overwritten
        raise NotImplementedError


class AccessOm2Builder(BaseBuilder):
    """Intake-esm catalog builder for ACCESS-OM2 COSIMA datasets"""

    def __init__(self, path):
        """
        Initialise a AccessOm2Builder

        Parameters
        ----------
        path : str or list of str
            Path or list of paths to crawl for assets/files.
        """

        kwargs = dict(
            path=path,
            depth=3,
            exclude_patterns=["*restart*", "*o2i.nc"],
            include_patterns=["*.nc"],
            data_format="netcdf",
            groupby_attrs=["file_id", "frequency"],
            aggregations=[
                {
                    "type": "join_existing",
                    "attribute_name": "start_date",
                    "options": {
                        "dim": "time",
                        "combine": "by_coords",
                    },
                },
            ],
        )

        super().__init__(**kwargs)

    @staticmethod
    def parser(file):
        try:
            match_groups = re.match(
                r".*/([^/]*)/([^/]*)/output\d+/([^/]*)/.*\.nc", file
            ).groups()
            # configuration = match_groups[0]
            # exp_id = match_groups[1]
            realm = match_groups[2]

            if realm == "ice":
                realm = "seaIce"

            filename = Path(file).stem

            # Get file id without any dates
            # - ocean-3d-v-1-monthly-pow02-ym_1958_04.nc
            # - iceh.057-daily.nc
            # - oceanbgc-3d-caco3-1-yearly-mean-y_2015.nc
            file_id = strip_pattern_rh(
                [
                    r"\d{4}[-_]\d{2}[-_]\d{2}",
                    r"\d{4}[-_]\d{2}",
                    r"\d{4}",
                    r"\d{3}",
                    r"\d{2}",
                ],
                filename,
            )

            with xr.open_dataset(
                file,
                chunks={},
                decode_cf=False,
                decode_times=False,
                decode_coords=False,
            ) as ds:
                variable_list = []
                variable_long_name_list = []
                variable_standard_name_list = []
                variable_cell_methods_list = []
                for var in ds.data_vars:
                    attrs = ds[var].attrs
                    if "long_name" in attrs:
                        variable_list.append(var)
                        variable_long_name_list.append(attrs["long_name"])
                    if "standard_name" in attrs:
                        variable_standard_name_list.append(attrs["standard_name"])
                    if "cell_methods" in attrs:
                        variable_cell_methods_list.append(attrs["cell_methods"])

                start_date, end_date, frequency = get_timeinfo(ds)

            info = {
                "path": str(file),
                "realm": realm,
                "variable": variable_list,
                "frequency": frequency,
                "start_date": start_date,
                "end_date": end_date,
                "variable_long_name": variable_long_name_list,
                "variable_standard_name": variable_standard_name_list,
                "variable_cell_methods": variable_cell_methods_list,
                "filename": filename,
                "file_id": file_id,
            }

            return info

        except Exception:
            return {INVALID_ASSET: file, TRACEBACK: traceback.format_exc()}


class AccessEsm15Builder(BaseBuilder):
    """Intake-esm catalog builder for ACCESS-ESM1.5 datasets"""

    def __init__(self, path, ensemble=False):
        """
        Initialise a AccessEsm15Builder

        Parameters
        ----------
        path: str or list of str
            Path or list of paths to crawl for assets/files.
        ensemble: boolean, optional
            Whether to treat each path as a separate member of an ensemble to join
            along a new member dimension
        """

        kwargs = dict(
            path=path,
            depth=3,
            exclude_patterns=["*restart*"],
            include_patterns=["*.nc*"],
            data_format="netcdf",
            groupby_attrs=["file_id", "frequency"],
            aggregations=[
                {
                    "type": "join_existing",
                    "attribute_name": "start_date",
                    "options": {
                        "dim": "time",
                        "combine": "by_coords",
                    },
                },
            ],
        )

        if ensemble:
            kwargs["aggregations"] += [
                {
                    "type": "join_new",
                    "attribute_name": "member",
                },
            ]

        super().__init__(**kwargs)

    @staticmethod
    def parser(file):
        try:
            match_groups = re.match(r".*/([^/]*)/history/([^/]*)/.*\.nc", file).groups()
            exp_id = match_groups[0]
            realm = match_groups[1]

            realm_mapping = {"atm": "atmos", "ocn": "ocean", "ice": "seaIce"}
            realm = realm_mapping[realm]

            filename = Path(file).stem

            # Get file id without any dates or exp_id
            # - iceh_m.2014-06.nc
            # - bz687a.pm107912_mon.nc
            # - bz687a.p7107912_mon.nc
            # - PI-GWL-B2035.pe-109904_dai.nc
            # - PI-1pct-02.pe-011802_dai.nc_dai.nc
            # - ocean_daily.nc-02531231
            file_id = strip_pattern_rh(
                [r"\d{4}[-_]\d{2}", r"\d{8}", r"\d{6}"], filename
            )
            file_id = strip_pattern_rh([exp_id], file_id)

            with xr.open_dataset(
                file,
                chunks={},
                decode_cf=False,
                decode_times=False,
                decode_coords=False,
            ) as ds:
                variable_list = []
                variable_long_name_list = []
                variable_standard_name_list = []
                variable_cell_methods_list = []
                for var in ds.data_vars:
                    attrs = ds[var].attrs
                    if "long_name" in attrs:
                        variable_list.append(var)
                        variable_long_name_list.append(attrs["long_name"])
                    if "standard_name" in attrs:
                        variable_standard_name_list.append(attrs["standard_name"])
                    if "cell_methods" in attrs:
                        variable_cell_methods_list.append(attrs["cell_methods"])

                start_date, end_date, frequency = get_timeinfo(ds)

            info = {
                "path": str(file),
                "realm": realm,
                "variable": variable_list,
                "frequency": frequency,
                "start_date": start_date,
                "end_date": end_date,
                "member": exp_id,
                "variable_long_name": variable_long_name_list,
                "variable_standard_name": variable_standard_name_list,
                "variable_cell_methods": variable_cell_methods_list,
                "filename": filename,
                "file_id": file_id,
            }

            return info

        except Exception:
            return {INVALID_ASSET: file, TRACEBACK: traceback.format_exc()}


# Include this so it is in the documentation
class AccessCm2Builder(AccessEsm15Builder):
    """Intake-esm catalog builder for ACCESS-CM2 datasets"""

    pass
