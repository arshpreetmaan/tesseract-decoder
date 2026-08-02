#ifndef GARI_TWO_STAGE_TESSERACT_H
#define GARI_TWO_STAGE_TESSERACT_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "stim.h"
#include "tesseract.h"
#include "utils.h"

// Describes the block boundaries in a monolithic GARI DEM. Detector rows are
// [physical, virtual], and error columns are [physical, barred]. Mapping values
// use the monolithic detector indices, before virtual rows are rebased.
struct GariTwoStageLayout {
  size_t physical_detector_count = 0;
  size_t virtual_detector_count = 0;
  size_t physical_error_count = 0;
  size_t barred_error_count = 0;
  std::vector<size_t> barred_error_to_virtual_detector;
};

struct GariTwoStageConfig {
  GariTwoStageLayout layout;
  size_t max_top_beam = 20;
  bool top_beam_climbing = false;
  size_t num_top_detector_orders = 21;
  size_t top_candidates_per_trial = 1;
  DetOrder top_detector_order_method = DetOrder::DetIndex;
  uint64_t top_detector_order_seed = 0;
  bool top_no_revisit_dets = false;
  size_t bottom_beam = 2;
  size_t pqlimit = DEFAULT_PQLIMIT;
  double top_det_penalty = 0;
  bool top_sparsify_errors = false;
  int top_sparsify_base_degree = -1;
  int top_sparsify_max_degree = -1;
  int top_sparsify_reactivate_limit = -1;
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
};

// Runs a fresh top search at every beam/order trial, then completes each top
// candidate using the physical-error model. The two child decoders and their
// graph data are retained across shots; the debt cache is intentionally per
// shot.
class GariTwoStageTesseractDecoder {
 public:
  GariTwoStageTesseractDecoder(const stim::DetectorErrorModel& gari_dem,
                               GariTwoStageConfig config);

  GariTwoStageDecodeResult decode(const std::vector<uint64_t>& top_detections);

  const stim::DetectorErrorModel& top_dem() const {
    return top_dem_;
  }
  const stim::DetectorErrorModel& bottom_dem() const {
    return bottom_dem_;
  }
  bool top_no_revisit_dets_enabled() const {
    return top_decoder_->config.no_revisit_dets;
  }
  bool bottom_no_revisit_dets_enabled() const {
    return bottom_decoder_->config.no_revisit_dets;
  }
  int top_sparsify_reactivate_limit() const {
    return top_decoder_->config.sparsify_reactivate_limit;
  }

 private:
  GariTwoStageConfig config_;
  stim::DetectorErrorModel top_dem_;
  stim::DetectorErrorModel bottom_dem_;
  std::vector<size_t> top_error_to_bottom_detector_;
  std::vector<std::vector<size_t>> top_error_detectors_;
  std::vector<std::vector<size_t>> bottom_error_detectors_;
  std::unique_ptr<TesseractDecoder> top_decoder_;
  std::unique_ptr<TesseractDecoder> bottom_decoder_;
};

#endif  // GARI_TWO_STAGE_TESSERACT_H
