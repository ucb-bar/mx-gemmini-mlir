# Share only completed artifacts with identical fresh bytes

Optional session weight-object sharing reduces build-storage pressure. The
selected assembler still produces a fresh object normally. Merlin reopens the
complete fresh and previous bytes, checks extent and SHA256 identity, then
atomically shares storage. Different bytes, symbolic paths and changed files
refuse. Providers must keep both completed files immutable. This supplies no
compiler, numeric, namespace, runtime-alias or executable-admission permission.

The target linker exposes `--reuse-linked-weights-object previous.o`, records
the actual fresh producer command/tool identity and raw input hash, and keeps
the ordinary final recipe and linked-image admission. Defaults are unchanged.
This is file storage reuse, not runtime encoded-weight deduplication or a model
speedup.

A complete two-program fixture reuses the same compiled stages and executes
one prefill plus three recurrent decodes. Its 128-byte constant pool produces
a 744-byte object. The new final executable is **byte-identical** to the
original admitted executable. All eight output/state hashes and four complete
2,048-byte output dumps match. Reset, preparation and step timers are identical:

| Inclusive work | Original cycles | Shared-object cycles |
|---|---:|---:|
| Reset | 1,330 | 1,330 |
| Prefill plus preparation | 160,739 | 160,739 |
| Three decodes plus preparation | 478,159 | 478,159 |
| Total | 640,228 | 640,228 |

A full cached-model link also completed fresh production and admission with a
262,513,256-byte weight object shared after the complete fresh-byte check.
Its 262,512,640 raw weight bytes retain SHA256
`fe29fc5c8f1281f5e512bc06c605d558a7ac5da32a218738e3fa38af8bfea04d`.
The actual 2,439,169,264-byte final image was rehashed against its recipe and
admission. Its complete model execution is still running: this is link/storage
evidence, without a completed runtime or numerical result for that image.

The existing session linker already shares the unchanged prefill catalog.
Additional task-local sharing of three completed decoder host blobs/objects
reopened both original recipes and all bytes, sharing 1,575,077,688 logical
bytes. Original file contents, recipes, model images and timing boundaries
remain unchanged. These checks provided a concrete storage budget for the full
cached compact-input evaluation; no cleanup outside the task was performed.

```sh
python docs/evidence/linked_weight_storage_20261019/verify.py
# Recheck actual shared inodes while the original files remain available:
python docs/evidence/linked_weight_storage_20261019/verify.py --check-live-storage
```

The standalone stdlib verifier first checks the pinned
[original guarded-session archive](../guarded_session_link_20261019/README.md),
then reopens all 140 files in this 69,044-byte archive. It checks both identical
small final images, admissions, completed recipes, actual log protocol, eight
output/state hashes, all 2,048 F32 dump words, original timers, producer and
weight-byte identities. The large admitted link/storage observation is retained
separately from its pending execution. Its large image/object payloads and
toolchain executables remain external; this is not a hermetic rebuild.

Source commits are Merlin `6599207c8` and MX `2bca820`. Twenty-six generic
artifact/constant/namespace tests and 30 session integration tests pass; core
Ruff/package-doc/structure/docs and target Ruff checks pass. The small fixture's
exact link/run command arrays are retained under `candidate/`. Reproduce the
original fixture and add the optional sharing flag to its guarded link using
the previous completed `weights_blob.o`; use a fresh task-local output directory.
