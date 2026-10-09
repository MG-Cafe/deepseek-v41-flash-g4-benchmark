# ==============================================================================
# Dockerfile — DeepSeek-V4.1-Flash on G4 (NVIDIA RTX PRO 6000 Blackwell SM120)
# 1M Context Window + DSpark Speculative Decoding + Sparse-MLA + MXFP4 Indexer
# ==============================================================================
ARG BASE_IMAGE=us-docker.pkg.dev/agent-platform-mg-public/containers/sglang-airlock:ds41-e56358a-rev1
FROM ${BASE_IMAGE}

WORKDIR /sgl-workspace/sglang

# Copy pre-patched SGLang SM120 base files & kernel optimization diffs
COPY sglang_overrides/ /sgl-workspace/sglang/python/sglang/
COPY patches/ /opt/dsv41_g4_patches/
COPY scripts/vertex_g4_entrypoint.sh /usr/local/bin/vertex_g4_entrypoint.sh

RUN chmod +x /usr/local/bin/vertex_g4_entrypoint.sh && \
    cd /sgl-workspace/sglang && \
    for p in \
      22-mhc-mix-stats-tf32-gated.diff \
      26-mhc-mix-stats-gather-pretrans-swizzle.diff \
      36-sparse-prefill-qtile.diff \
      36b-sparse-prefill-v2.diff \
      37-mhc-post-pre-fuse.diff \
      38-sparse-decode-splitk.diff \
      41-prefill-shard-mhc.diff \
      44-prefill-attn-rowshard.diff \
      45-sparse-prefill-h64.diff \
      46-fp4-gemm-direct.diff \
      48-indexer-mask-incr.diff; do \
      if [ -f "/opt/dsv41_g4_patches/$p" ]; then \
        patch -p1 --forward < "/opt/dsv41_g4_patches/$p" || true; \
      fi; \
    done && \
    find /sgl-workspace/sglang/python/sglang -name '__pycache__' -prune -exec rm -rf {} +

EXPOSE 7080
ENTRYPOINT ["/usr/local/bin/vertex_g4_entrypoint.sh"]
