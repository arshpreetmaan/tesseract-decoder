#ifndef GARI_TWO_STAGE_TESSERACT_H
#define GARI_TWO_STAGE_TESSERACT_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "stim.h"
#include "tesseract.h"
#include "utils.h"

// Describes the block boundaries in a monolithic GARI DEM. Detector rows are
// [physical, virtual], and error columns are [physical, barred]. Mapping values
// use the monolithic detector indices, before virtual rows are rebased.
struct GariBlockRange {
  size_t offset = 0;
  size_t count = 0;
};

struct GariTopComponentLayout {
  GariBlockRange detector_rows;
  GariBlockRange barred_error_columns;
  GariBlockRange debt_detector_rows;
};

struct GariTopComponentsLayout {
  GariTopComponentLayout d_x;
  GariTopComponentLayout d_z;
};

struct GariTwoStageLayout {
  size_t physical_detector_count = 0;
  size_t virtual_detector_count = 0;
  size_t physical_error_count = 0;
  size_t barred_error_count = 0;
  std::vector<size_t> barred_error_to_virtual_detector;
  std::optional<GariTopComponentsLayout> top_components;
};

struct GariPreparedTopComponent {
  stim::DetectorErrorModel dem;
  std::vector<size_t> error_to_top_error;
  std::vector<std::vector<size_t>> error_detectors;
};

// Immutable result of validating and partitioning one monolithic GARI DEM.
// This is prepared once and shared by the per-thread decoder instances.
struct GariTwoStagePreparedModel {
  GariTwoStageLayout layout;
  stim::DetectorErrorModel top_dem;
  stim::DetectorErrorModel bottom_dem;
  std::vector<size_t> top_error_to_bottom_detector;
  std::vector<std::vector<size_t>> top_error_detectors;
  std::vector<std::vector<size_t>> bottom_error_detectors;

  // Populated before workers when the direct-sum D_X/D_Z split is requested.
  GariPreparedTopComponent d_x_top;
  GariPreparedTopComponent d_z_top;
};

std::shared_ptr<const GariTwoStagePreparedModel> prepare_gari_two_stage_model(
    const stim::DetectorErrorModel& gari_dem, const GariTwoStageLayout& layout,
    bool prepare_top_components = false);

struct GariTwoStageConfig {
  size_t max_top_beam = 20;
  bool top_beam_climbing = false;
  bool split_top = false;
  size_t num_top_detector_orders = 21;
  size_t top_candidates_per_trial = 1;
  DetOrder top_detector_order_method = DetOrder::DetIndex;
  uint64_t top_detector_order_seed = 0;
  bool top_no_revisit_dets = false;
  size_t bottom_beam = 2;
  size_t num_bottom_detector_orders = 1;
  size_t pqlimit = DEFAULT_PQLIMIT;
  double top_det_penalty = 0;
  bool top_sparsify_errors = false;
  int top_sparsify_base_degree = -1;
  int top_sparsify_max_degree = -1;
  int top_sparsify_reactivate_limit = -1;
  bool collect_bottom_timing = false;
  bool verbose = false;
  std::vector<std::vector<size_t>> top_detector_orders;
  // Source detector index -> top GARI detector index. DetIndex orders follow
  // this source chronology instead of the block-grouped GARI row order.
  std::vector<size_t> source_to_top_detector;
};

struct GariTwoStageDecodeResult {
  bool completed = false;
  double physical_cost = std::numeric_limits<double>::infinity();
  std::vector<int> observables;
  std::vector<size_t> top_errors;
  std::vector<size_t> physical_errors;

  size_t unique_debts = 0;
  size_t bottom_cache_hits = 0;
  double bottom_decode_time_seconds = 0;
};

// Runs a fresh top search at every beam/order trial, then completes each top
// candidate using the physical-error model. Child decoder graph data is
// retained across shots; the debt cache is intentionally per shot.
class GariTwoStageTesseractDecoder {
 public:
  GariTwoStageTesseractDecoder(
      std::shared_ptr<const GariTwoStagePreparedModel> prepared_model,
      GariTwoStageConfig config);

  GariTwoStageDecodeResult decode(const std::vector<uint64_t>& top_detections);

  const stim::DetectorErrorModel& top_dem() const {
    return prepared_model_->top_dem;
  }
  const stim::DetectorErrorModel& bottom_dem() const {
    return prepared_model_->bottom_dem;
  }
  const std::vector<std::vector<size_t>>& bottom_detector_orders() const {
    return bottom_decoder_->config.det_orders;
  }
  const std::vector<std::vector<size_t>>& d_x_top_detector_orders() const {
    return d_x_top_decoder_->config.det_orders;
  }
  const std::vector<std::vector<size_t>>& d_z_top_detector_orders() const {
    return d_z_top_decoder_->config.det_orders;
  }
  bool top_no_revisit_dets_enabled() const {
    return active_top_decoder().config.no_revisit_dets;
  }
  bool bottom_no_revisit_dets_enabled() const {
    return bottom_decoder_->config.no_revisit_dets;
  }
  int top_sparsify_reactivate_limit() const {
    return active_top_decoder().config.sparsify_reactivate_limit;
  }

 private:
  const TesseractDecoder& active_top_decoder() const;

  std::shared_ptr<const GariTwoStagePreparedModel> prepared_model_;
  GariTwoStageConfig config_;
  std::unique_ptr<TesseractDecoder> top_decoder_;
  std::unique_ptr<TesseractDecoder> d_x_top_decoder_;
  std::unique_ptr<TesseractDecoder> d_z_top_decoder_;
  std::unique_ptr<TesseractDecoder> bottom_decoder_;
};

#endif  // GARI_TWO_STAGE_TESSERACT_H
