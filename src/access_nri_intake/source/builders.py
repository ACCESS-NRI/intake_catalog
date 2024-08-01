# Copyright 2023 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

""" Builders for generating Intake-ESM datastores """

import multiprocessing
import re
import traceback
from pathlib import Path
import xarray as xr

from ecgtools.builder import INVALID_ASSET, TRACEBACK, Builder

from ..utils import validate_against_schema
from . import ESM_JSONSCHEMA, PATH_COLUMN, VARIABLE_COLUMN
from .utils import PATTERNS, FREQUENCIES, EmptyFileError, get_timeinfo


class ParserError(Exception):
    pass


class BaseBuilder(Builder):
    """
    Base class for creating Intake-ESM datastore builders. Not intended for direct use.
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

        super().__post_init__()

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
        Save datastore contents to a file.

        Parameters
        ----------
        name: str
            The name of the file to save the datastore to.
        description : str
            Detailed multi-line description of the collection.
        directory: str, optional
            The directory to save the datastore to. If None, use the current directory.
        """

        if self.df.empty:
            raise ValueError(
                "Intake-ESM datastore has not yet been built. Please run `.build()` first"
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
        Builds a datastore from a list of netCDF files or zarr stores.
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
            .map(type)
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

    @staticmethod
    def parse_access_filename(
        filename, 
        patterns=PATTERNS, 
        frequencies=FREQUENCIES, 
        redaction_fill : str = "X"
    ):
        """
        Parse an ACCESS model filename and return a file id and any time information

        Parameters
        ----------
        filename: str
            The filename to parse with the extension removed

        Returns
        -------
        file_id: str
            The file id constructed by redacting time information and replacing non-python characters
            with underscores
        timestamp: str
            A string of the redacted time information (e.g. "1990-01")
        frequency: str
            The frequency of the file if available in the filename
        """

        # Try to determine frequency
        frequency = None
        for pattern, freq in frequencies.items():
            if re.search(pattern, filename):
                frequency = freq
                break

        # Parse file id
        file_id = filename
        timestamp = None
        for pattern in patterns:
            match = re.match(pattern, file_id)
            if match:
                timestamp = match.group(1)
                redaction = re.sub(r"\d", redaction_fill, timestamp)
                file_id = file_id[: match.start(1)] + redaction + file_id[match.end(1) :]
                break

        # Remove non-python characters from file ids
        file_id = re.sub(r"[-.]", "_", file_id)
        file_id = re.sub(r"_+", "_", file_id).strip("_")

        return file_id, timestamp, frequency

    @classmethod
    def parse_access_ncfile(cls, file, time_dim="time"):
        """
        Get Intake-ESM datastore entry info from an ACCESS netcdf file

        Parameters
        ----------
        file: str
            The path to the netcdf file
        time_dim: str
            The name of the time dimension

        Returns
        -------
        """

        file = Path(file)
        filename = file.name

        file_id, filename_timestamp, filename_frequency = cls.parse_access_filename(file.stem)

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
            variable_units_list = []
            for var in ds.data_vars:
                attrs = ds[var].attrs
                if "long_name" in attrs:
                    variable_list.append(var)
                    variable_long_name_list.append(attrs["long_name"])
                    if "standard_name" in attrs:
                        variable_standard_name_list.append(attrs["standard_name"])
                    else:
                        variable_standard_name_list.append("")
                    if "cell_methods" in attrs:
                        variable_cell_methods_list.append(attrs["cell_methods"])
                    else:
                        variable_cell_methods_list.append("")
                    if "units" in attrs:
                        variable_units_list.append(attrs["units"])
                    else:
                        variable_units_list.append("")

            start_date, end_date, frequency = get_timeinfo(ds, filename_frequency, time_dim)

        if not variable_list:
            raise EmptyFileError("This file contains no variables")

        outputs = (
            filename,
            file_id,
            filename_timestamp,
            frequency,
            start_date,
            end_date,
            variable_list,
            variable_long_name_list,
            variable_standard_name_list,
            variable_cell_methods_list,
            variable_units_list,
        )

        return outputs


class AccessOm2Builder(BaseBuilder):
    """Intake-ESM datastore builder for ACCESS-OM2 COSIMA datasets"""

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

    @classmethod
    def parser(cls, file):
        try:
            match_groups = re.match(r".*/output\d+/([^/]*)/.*\.nc", file).groups()
            realm = match_groups[0]

            if realm == "ice":
                realm = "seaIce"

            (
                filename,
                file_id,
                _,
                frequency,
                start_date,
                end_date,
                variable_list,
                variable_long_name_list,
                variable_standard_name_list,
                variable_cell_methods_list,
                variable_units_list,
            ) = cls.parse_access_ncfile(file)

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
                "variable_units": variable_units_list,
                "filename": filename,
                "file_id": file_id,
            }

            return info

        except Exception:
            return {INVALID_ASSET: file, TRACEBACK: traceback.format_exc()}


class AccessOm3Builder(BaseBuilder):
    """Intake-ESM datastore builder for ACCESS-OM3 COSIMA datasets"""

    def __init__(self, path):
        """
        Initialise a AccessOm3Builder

        Parameters
        ----------
        path : str or list of str
            Path or list of paths to crawl for assets/files.
        """

        kwargs = dict(
            path=path,
            depth=2,
            exclude_patterns=[
                "*restart*",
                "*MOM_IC.nc",
                "*ocean_geometry.nc",
                "*ocean.stats.nc",
                "*Vertical_coordinate.nc",
            ],
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

    @classmethod
    def parser(cls, file):
        try:
            (
                filename,
                file_id,
                _,
                frequency,
                start_date,
                end_date,
                variable_list,
                variable_long_name_list,
                variable_standard_name_list,
                variable_cell_methods_list,
                variable_units_list,
            ) = cls.parse_access_ncfile(file)

            if "mom6" in filename:
                realm = "ocean"
            elif "ww3" in filename:
                realm = "wave"
            elif "cice" in filename:
                realm = "seaIce"
            else:
                raise ParserError(f"Cannot determine realm for file {file}")

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
                "variable_units": variable_units_list,
                "filename": filename,
                "file_id": file_id,
            }

            return info

        except Exception:
            return {INVALID_ASSET: file, TRACEBACK: traceback.format_exc()}


class AccessEsm15Builder(BaseBuilder):
    """Intake-ESM datastore builder for ACCESS-ESM1.5 datasets"""

    def __init__(self, path, ensemble):
        """
        Initialise a AccessEsm15Builder

        Parameters
        ----------
        path: str or list of str
            Path or list of paths to crawl for assets/files.
        ensemble: boolean
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

    @classmethod
    def parser(cls, file):
        try:
            match_groups = re.match(r".*/([^/]*)/history/([^/]*)/.*\.nc", file).groups()
            exp_id = match_groups[0]
            realm = match_groups[1]

            realm_mapping = {"atm": "atmos", "ocn": "ocean", "ice": "seaIce"}
            realm = realm_mapping[realm]

            (
                filename,
                file_id,
                _,
                frequency,
                start_date,
                end_date,
                variable_list,
                variable_long_name_list,
                variable_standard_name_list,
                variable_cell_methods_list,
                variable_units_list,
            ) = cls.parse_access_ncfile(file)

            # Remove exp_id from file id so that members can be part of the same dataset
            file_id = re.sub(exp_id, "", file_id).strip("_")

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
                "variable_units": variable_units_list,
                "filename": filename,
                "file_id": file_id,
            }

            return info

        except Exception:
            return {INVALID_ASSET: file, TRACEBACK: traceback.format_exc()}


# Include this so it is in the documentation
class AccessCm2Builder(AccessEsm15Builder):
    """Intake-ESM datastore builder for ACCESS-CM2 datasets"""

    pass
