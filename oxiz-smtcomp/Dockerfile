# syntax=docker/dockerfile:1.7
# Multi-stage build for oxiz-smtcomp runner (smtcomp2026 binary).
#
# Build (from the repo root):
#   docker build -f oxiz-smtcomp/Dockerfile -t oxiz-smtcomp:latest .
#
# Run a single SMT-LIB2 file (the binary takes an optional FILE argument
# or reads from stdin). Exit codes: 0=sat, 10=unsat, 20=unknown/error.
#   docker run --rm -v "$PWD/bench:/bench:ro" \
#     oxiz-smtcomp:latest /bench/problem.smt2
#
# Or pipe via stdin:
#   cat problem.smt2 | docker run --rm -i oxiz-smtcomp:latest
#
# Print version:
#   docker run --rm oxiz-smtcomp:latest --version
#
# Pin the Rust toolchain via build arg (default: current stable that
# supports edition 2024). Override with:
#   docker build --build-arg RUST_VERSION=1.89 -f oxiz-smtcomp/Dockerfile -t oxiz-smtcomp:latest .

ARG RUST_VERSION=1.85

FROM rust:${RUST_VERSION}-slim-bookworm AS builder

WORKDIR /workspace

# Pure-Rust build: no native linker deps beyond the default toolchain.
# ca-certificates is kept so `cargo` can reach crates.io over HTTPS
# when the local Cargo.lock references registry packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy the workspace manifests and sources. Every workspace member's
# Cargo.toml must be present for cargo to resolve the workspace even
# when building a single package; `oxiz-py` is included for that reason
# (it is not built by this Dockerfile).
COPY Cargo.toml Cargo.lock ./
COPY oxiz ./oxiz
COPY oxiz-core ./oxiz-core
COPY oxiz-math ./oxiz-math
COPY oxiz-sat ./oxiz-sat
COPY oxiz-nlsat ./oxiz-nlsat
COPY oxiz-theories ./oxiz-theories
COPY oxiz-solver ./oxiz-solver
COPY oxiz-opt ./oxiz-opt
COPY oxiz-spacer ./oxiz-spacer
COPY oxiz-proof ./oxiz-proof
COPY oxiz-wasm ./oxiz-wasm
COPY oxiz-cli ./oxiz-cli
COPY oxiz-smtcomp ./oxiz-smtcomp
COPY oxiz-py ./oxiz-py
COPY oxiz-ml ./oxiz-ml
COPY bench ./bench

# Build the release binary with the workspace release profile
# (opt-level=z, lto, panic=abort, strip=symbols).
RUN cargo build --release --package oxiz-smtcomp --bin smtcomp2026

# ---- Runtime ----
FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /workspace/target/release/smtcomp2026 /usr/local/bin/smtcomp2026
COPY oxiz-smtcomp/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/smtcomp2026

# Non-root user for safety.
RUN useradd --system --no-create-home --shell /usr/sbin/nologin oxiz
USER oxiz

WORKDIR /workspace
VOLUME ["/bench", "/results"]

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["--help"]
