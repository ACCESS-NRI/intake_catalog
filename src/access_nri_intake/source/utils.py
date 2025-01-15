# Copyright 2023 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for writing Intake-ESM builders and their parsers"""

import warnings
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

import cftime
import xarray as xr

FREQUENCY_STATIC = "fx"


class EmptyFileError(Exception):
    pass


@dataclass
class _NCFileInfo:
    """
    Holds information about a NetCDF file that is used to create an intake-esm
    catalog entry.

    Notes
    -----
    Use of both path and filename seems redundant, but constructing filename from
    the path using a __post_init__ method makes testing more difficult. On balance,
    more explicit tests are probably more important than the slight redundancy.
    """

    filename: str | Path
    file_id: str
    path: str
    filename_timestamp: str | None
    frequency: str
    start_date: str
    end_date: str
    variable: list[str]
    variable_long_name: list[str]
    variable_standard_name: list[str]
    variable_cell_methods: list[str]
    variable_units: list[str]

    def to_dict(self) -> dict[str, str | list[str]]:
        """
        Return a dictionary representation of the NcFileInfo object
        """
        return asdict(self)


@dataclass
class _VarInfo:
    """
    Holds information about the variables in a NetCDF file that is used to
    create an intake-esm catalog entry.
    """

    variable_list: list[str] = field(default_factory=list)
    long_name_list: list[str] = field(default_factory=list)
    standard_name_list: list[str] = field(default_factory=list)
    cell_methods_list: list[str] = field(default_factory=list)
    units_list: list[str] = field(default_factory=list)

    def append_attrs(self, var: str, attrs: dict) -> None:
        """
        Append attributes to the _VarInfo object, if the attribute has a
        'long_name' key.
        """
        if "long_name" not in attrs:
            return None

        self.variable_list.append(var)
        self.long_name_list.append(attrs["long_name"])
        self.standard_name_list.append(attrs.get("standard_name", ""))
        self.cell_methods_list.append(attrs.get("cell_methods", ""))
        self.units_list.append(attrs.get("units", ""))

    def to_var_info_dict(self) -> dict[str, list[str]]:
        """
        Return a dictionary representation of the _VarInfo object. Fields are
        defined explicitly for use in the _AccessNCFileInfo constructor.
        """
        return {
            "variable": self.variable_list,
            "variable_long_name": self.long_name_list,
            "variable_standard_name": self.standard_name_list,
            "variable_cell_methods": self.cell_methods_list,
            "variable_units": self.units_list,
        }


def _add_month_start(time, n: int):
    """Add months to cftime datetime and truncate to start"""
    year = time.year + ((time.month + n - 1) // 12)
    month = (time.month + n - 1) % 12 + 1
    return time.replace(
        year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0
    )


def _add_year_start(time, n: int):
    """Add years to cftime datetime and truncate to start"""
    return time.replace(
        year=time.year + n, month=1, day=1, hour=0, minute=0, second=0, microsecond=0
    )


def _guess_start_end_dates(ts, te, frequency):
    """Guess the start and end bounded times for a given frequency"""
    warnings.warn(
        "Time coordinate does not include bounds information. Guessing "
        "start and end times."
    )
    num, unit = frequency
    if unit == "yr":
        step_back = -int(num / 2)
        step_fwd = num + step_back
        ts = _add_year_start(ts, step_back)
        te = _add_year_start(te, step_fwd)
    elif unit == "mon":
        step_back = -int(num / 2)
        step_fwd = num + step_back
        ts = _add_month_start(ts, step_back)
        te = _add_month_start(te, step_fwd)
    elif unit == "day":
        dt = timedelta(days=num) / 2
        ts = ts - dt
        te = te + dt
    elif unit == "hr":
        dt = timedelta(hours=num) / 2
        ts = ts - dt
        te = te + dt
    else:
        warnings.warn("Cannot infer start and end times for subhourly frequencies.")
    return ts, te


class GenericTimeParser:
    """
    Generic time parser
    """

    def __init__(self, ds: xr.Dataset, filename_frequency: str | None, time_dim: str):
        """
        Parameters
        ----------
        ds: :py:class:`xarray.Dataset`
            The dataset to parse the time info from
        filename_frequency: str
            Frequency as determined from the filename
        time_dim: str
            The name of the time dimension
        """
        self.ds = ds
        self.filename_frequency = filename_frequency
        self.time_dim = time_dim

    @staticmethod
    def _add_month_start(time, n: int):
        """Add months to cftime datetime and truncate to start"""
        year = time.year + ((time.month + n - 1) // 12)
        month = (time.month + n - 1) % 12 + 1
        return time.replace(
            year=year, month=month, day=1, hour=0, minute=0, second=0, microsecond=0
        )

    @staticmethod
    def _add_year_start(time, n: int):
        """Add years to cftime datetime and truncate to start"""
        return time.replace(
            year=time.year + n,
            month=1,
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )

    @staticmethod
    def _guess_start_end_dates(ts, te, frequency):
        """Guess the start and end bounded times for a given frequency"""
        warnings.warn(
            "Time coordinate does not include bounds information. Guessing "
            "start and end times."
        )
        num, unit = frequency
        if unit == "yr":
            step_back = -int(num / 2)
            step_fwd = num + step_back
            ts = GenericTimeParser._add_year_start(ts, step_back)
            te = GenericTimeParser._add_year_start(te, step_fwd)
        elif unit == "mon":
            step_back = -int(num / 2)
            step_fwd = num + step_back
            ts = GenericTimeParser._add_month_start(ts, step_back)
            te = GenericTimeParser._add_month_start(te, step_fwd)
        elif unit == "day":
            dt = timedelta(days=num) / 2
            ts = ts - dt
            te = te + dt
        elif unit == "hr":
            dt = timedelta(hours=num) / 2
            ts = ts - dt
            te = te + dt
        else:
            warnings.warn("Cannot infer start and end times for subhourly frequencies.")
        return ts, te

    def _get_timeinfo(self) -> tuple[str, str, str]:
        """
        Get start date, end date and frequency of a xarray dataset. Stolen and adapted from the
        cosima cookbook, see
        https://github.com/COSIMA/cosima-cookbook/blob/master/cosima_cookbook/database.py#L565

        Parameters
        ----------
        ds: :py:class:`xarray.Dataset`
            The dataset to parse the time info from
        filename_frequency: str
            Frequency as determined from the filename
        time_dim: str
            The name of the time dimension

        Returns
        -------
        start_date: str
            The start date of the dataset
        end_date: str
            The end date of the dataset
        frequency: str
            The frequency of the dataset

        Raises
        ------
        EmptyFileError
            If the dataset has a valid unlimited dimension, but no data
        """

        ds = self.ds
        filename_frequency = self.filename_frequency
        time_dim = self.time_dim

        def _todate(t):
            return cftime.num2date(t, time_var.units, calendar=time_var.calendar)

        time_format = "%Y-%m-%d, %H:%M:%S"
        ts = None
        te = None
        frequency: str | tuple[int | None, str] = FREQUENCY_STATIC
        has_time = time_dim in ds

        if has_time:
            time_var = ds[time_dim]

            if len(time_var) == 0:
                raise EmptyFileError(
                    "This file has a valid unlimited dimension, but no data"
                )

            has_bounds = hasattr(time_var, "bounds") and time_var.bounds in ds.variables
            if has_bounds:
                bounds_var = ds.variables[time_var.bounds]
                ts = _todate(bounds_var[0, 0])
                te = _todate(bounds_var[-1, 1])
            else:
                ts = _todate(time_var[0])
                te = _todate(time_var[-1])

            if len(time_var) > 1 or has_bounds:
                if has_bounds:
                    t1 = _todate(bounds_var[0, 1])
                else:
                    t1 = _todate(time_var[1])

                dt = t1 - ts
                # TODO: This is not a very good way to get the frequency
                if dt.days >= 365:
                    years = round(dt.days / 365)
                    frequency = (years, "yr")
                elif dt.days >= 28:
                    months = round(dt.days / 30)
                    frequency = (months, "mon")
                elif dt.days >= 1:
                    frequency = (dt.days, "day")
                elif dt.seconds >= 3600:
                    hours = round(dt.seconds / 3600)
                    frequency = (hours, "hr")
                else:
                    frequency = (None, "subhr")

        if filename_frequency:
            if filename_frequency != frequency:
                msg = (
                    f"The frequency '{filename_frequency}' determined from filename does not "
                    f"match the frequency '{frequency}' determined from the file contents."
                )
                if frequency == FREQUENCY_STATIC:
                    frequency = filename_frequency
                warnings.warn(f"{msg} Using '{frequency}'.")

        if has_time & (frequency != FREQUENCY_STATIC):
            if not has_bounds:
                ts, te = GenericTimeParser._guess_start_end_dates(ts, te, frequency)

        if ts is None:
            start_date = "none"
        else:
            start_date = ts.strftime(time_format)

        if te is None:
            end_date = "none"
        else:
            end_date = te.strftime(time_format)

        if frequency[0]:
            frequency = f"{str(frequency[0])}{frequency[1]}"
        else:
            frequency = frequency[1]

        return start_date, end_date, frequency

    def __call__(self) -> tuple[str, str, str]:
        return self._get_timeinfo()


class AccessTimeParser(GenericTimeParser):
    pass


class GfdlTimeParser(GenericTimeParser):
    def __init__(self, ds: xr.Dataset, filename_frequency: str | None, time_dim: str):
        self.ds = ds
        self.filename_frequency = filename_frequency
        self.time_dim = time_dim

    def _get_timeinfo(self) -> tuple[str, str, str]:
        """
        Get start date, end date and frequency of a xarray dataset. Stolen and adapted from the
        cosima cookbook, see
        https://github.com/COSIMA/cosima-cookbook/blob/master/cosima_cookbook/database.py#L565

        Parameters
        ----------
        ds: :py:class:`xarray.Dataset`
            The dataset to parse the time info from
        filename_frequency: str
            Frequency as determined from the filename
        time_dim: str
            The name of the time dimension

        Returns
        -------
        start_date: str
            The start date of the dataset
        end_date: str
            The end date of the dataset
        frequency: str
            The frequency of the dataset

        Raises
        ------
        EmptyFileError
            If the dataset has a valid unlimited dimension, but no data
        """

        ds = self.ds
        filename_frequency = self.filename_frequency
        time_dim = self.time_dim

        def _todate(t):
            return cftime.num2date(t, time_var.units, calendar=time_var.calendar)

        time_format = "%Y-%m-%d, %H:%M:%S"
        ts = None
        te = None
        frequency: str | tuple[int | None, str] = FREQUENCY_STATIC
        has_time = time_dim in ds

        if has_time:
            time_var = ds[time_dim]

            if len(time_var) == 0:
                raise EmptyFileError(
                    "This file has a valid unlimited dimension, but no data"
                )

            ts = _todate(time_var[0])
            te = _todate(time_var[-1])

            if len(time_var) > 1:
                t1 = _todate(time_var[1])

                dt = t1 - ts
                # TODO: This is not a very good way to get the frequency
                if dt.days >= 365:
                    years = round(dt.days / 365)
                    frequency = (years, "yr")
                elif dt.days >= 28:
                    months = round(dt.days / 30)
                    frequency = (months, "mon")
                elif dt.days >= 1:
                    frequency = (dt.days, "day")
                elif dt.seconds >= 3600:
                    hours = round(dt.seconds / 3600)
                    frequency = (hours, "hr")
                else:
                    frequency = (None, "subhr")

        if filename_frequency:
            if filename_frequency != frequency:
                msg = (
                    f"The frequency '{filename_frequency}' determined from filename does not "
                    f"match the frequency '{frequency}' determined from the file contents."
                )
                if frequency == FREQUENCY_STATIC:
                    frequency = filename_frequency
                warnings.warn(f"{msg} Using '{frequency}'.")

        if has_time & (frequency != FREQUENCY_STATIC):
            ts, te = GenericTimeParser._guess_start_end_dates(ts, te, frequency)

        if ts is None:
            start_date = "none"
        else:
            start_date = ts.strftime(time_format)

        if te is None:
            end_date = "none"
        else:
            end_date = te.strftime(time_format)

        if frequency[0]:
            frequency = f"{str(frequency[0])}{frequency[1]}"
        else:
            frequency = frequency[1]

        return start_date, end_date, frequency

    def __call__(self) -> tuple[str, str, str]:
        return self._get_timeinfo()
