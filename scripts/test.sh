#!/usr/bin/env sh
# Run the test suite inside the image, against the pinned dependencies that
# actually ship. Nothing here touches the network or the real Hub.
#
#   ./scripts/test.sh              everything
#   ./scripts/test.sh -k storage   one slice
#   ./scripts/test.sh -x -q        stop at the first failure
set -eu

cd "$(dirname "$0")/.."

docker build -q -t trove:test . >/dev/null

# A thin layer on top of the app image adds pytest; Docker caches it, so only
# the first run pays for the install.
docker build -q -t trove:test-runner -f - . >/dev/null <<'DOCKERFILE'
FROM trove:test
COPY tests/requirements.txt /tmp/test-requirements.txt
RUN pip install --no-cache-dir -r /tmp/test-requirements.txt
DOCKERFILE

# app/ and tests/ are mounted rather than baked in, so a run picks up whatever
# is in the working tree without another image build.
exec docker run --rm \
  -v "$PWD/tests:/srv/tests:ro" \
  -v "$PWD/app:/srv/app:ro" \
  -v "$PWD/pytest.ini:/srv/pytest.ini:ro" \
  -w /srv \
  trove:test-runner \
  python -u -m pytest -p no:cacheprovider "$@"
