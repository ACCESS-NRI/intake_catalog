# Copyright 2023 ACCESS-NRI and contributors. See the top-level COPYRIGHT file for details.
# SPDX-License-Identifier: Apache-2.0

import warnings

import intake
import intake.catalog
from access_ipy_telemetry.api import ApiHandler
from access_ipy_telemetry.cli import configure_telemetry

from access_nri_intake.utils import get_catalog_fp

api_handler = ApiHandler()

CATALOG_NAME_FORMAT = (
    r"^v(?P<yr>2[0-9]{3})\-(?P<mon>1[0-2]|0[1-9])\-(?P<day>0[1-9]|[1-2][0-9]|3[0-1])$"
)


try:
    data = intake.open_catalog(get_catalog_fp()).access_nri
    cat_version = data._captured_init_kwargs.get("metadata", {}).get(
        "version", "latest"
    )  # Get the catalog version number and set it to "latest" if it can't be found
    configure_telemetry(["--enable", "--silent"])
    api_handler.add_extra_field("catalog", {"catalog_version": cat_version})
except FileNotFoundError:
    warnings.warn(
        "Unable to access a default catalog location. Calling intake.cat.access_nri will not work.",
        RuntimeWarning,
        stacklevel=2,
    )
    data = intake.catalog.Catalog()
