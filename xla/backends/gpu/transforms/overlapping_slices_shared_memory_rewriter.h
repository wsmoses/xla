#ifndef XLA_BACKENDS_GPU_TRANSFORMS_OVERLAPPING_SLICES_SHARED_MEMORY_REWRITER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_OVERLAPPING_SLICES_SHARED_MEMORY_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites `kLoop` fusions containing `concatenate` -> `dynamic_slice` subgraphs 
// into `kCustom` fusions with `__overlapping_slices` config.
class OverlappingSlicesSharedMemoryRewriter : public HloModulePass {
 public:
  explicit OverlappingSlicesSharedMemoryRewriter() = default;
  ~OverlappingSlicesSharedMemoryRewriter() override = default;

  absl::string_view name() const override {
    return "overlapping-slices-shared-memory-rewriter";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_OVERLAPPING_SLICES_SHARED_MEMORY_REWRITER_H_
