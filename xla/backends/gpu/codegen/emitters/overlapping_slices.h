#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_OVERLAPPING_SLICES_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_OVERLAPPING_SLICES_H_

#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla {
namespace gpu {

// Lowers `__overlapping_slices` custom fusion to LLVM via MLIR using GPU shared memory.
class OverlappingSlicesFusion : public EmitterBase {
 public:
  explicit OverlappingSlicesFusion(const HloFusionAnalysis& analysis);

  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* mlir_context) const override;

  std::optional<std::vector<IndexingMap>> ComputeThreadIdToInputIndexing(
      int64_t root_index, mlir::MLIRContext* mlir_context) const override;

 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

  std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const override;

 private:
  const HloFusionAnalysis& analysis_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_OVERLAPPING_SLICES_H_
