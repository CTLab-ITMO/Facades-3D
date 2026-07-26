#!/usr/bin/env bash

set -Eeuo pipefail

TMP_TRELLIS_DOWNLOAD_PATH="/tmp/temp-trellis-download"
git clone --depth 1 --filter=blob:none --sparse https://github.com/microsoft/TRELLIS.git $TMP_TRELLIS_DOWNLOAD_PATH

cd $TMP_TRELLIS_DOWNLOAD_PATH

git sparse-checkout set trellis/models/ trellis/modules/ trellis/pipelines/ trellis/renderers/ trellis/representations/ trellis/utils/
git submodule update --init --recursive -- trellis/representations/mesh/flexicubes/

cat <<EOF > trellis/__init__.py
from . import models
from . import modules
from . import pipelines
from . import renderers
from . import representations
from . import utils
EOF

cd $OLDPWD

mv $TMP_TRELLIS_DOWNLOAD_PATH/trellis/ src/trellis/
rm -rf $TMP_TRELLIS_DOWNLOAD_PATH

