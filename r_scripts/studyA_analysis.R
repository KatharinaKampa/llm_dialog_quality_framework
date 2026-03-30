# Study A (LLM Evaluation) - Preregistered Analyses H1-H9
# PRIMARY + SENS

# Packages
req_pkgs <- c(
  "tidyverse", "readr", "janitor", "boot",
  "pROC", "irr"
)
to_install <- req_pkgs[!req_pkgs %in% installed.packages()[, "Package"]]
if (length(to_install) > 0) install.packages(to_install)

library(tidyverse)
library(readr)
library(janitor)
library(boot)
library(pROC)
library(irr)

options(warn = 1)
set.seed(1234)

# File paths
PATH_HUMAN_PRIMARY_GOLD  <- "data/human_PRIMARY_gold_dialog_level.csv"
PATH_HUMAN_PRIMARY_ROWS  <- "data/human_PRIMARY_rows_regular_only_kept.csv"
PATH_HUMAN_SENS_GOLD     <- "data/human_SENS_gold_dialog_level.csv"
PATH_HUMAN_SENS_ROWS     <- "data/human_SENS_rows_regular_only_kept.csv"
PATH_LLM_PRIMARY_SUBSET  <- "data/llm_PRIMARY_median_subset.csv"
PATH_LLM_SENS_SUBSET     <- "data/llm_SENS_median_subset.csv"
PATH_LLM_ALL             <- "data/llm_median_all_dialogs.csv"
PATH_LLM_RUNS            <- "data/llm_runs_all_dialogs.csv"

# Helpers
stop_if_missing <- function(df, cols, name = "dataframe") {
  miss <- setdiff(cols, names(df))
  if (length(miss) > 0) {
    stop(sprintf("[%s] Missing required columns: %s", name, paste(miss, collapse = ", ")))
  }
}

as_good_bad <- function(x) {
  x <- tolower(trimws(as.character(x)))
  x <- ifelse(x %in% c("good", "g", "high", "1", "true"), "good",
              ifelse(x %in% c("bad", "b", "low", "0", "false"), "bad", x))
  factor(x, levels = c("bad", "good"))
}

# Percentile bootstrap CI from precomputed bootstrap statistics (prereg-conform)
boot_ci <- function(boot_stats, estimate, conf = 0.95) {
  x <- boot_stats[is.finite(boot_stats)]
  if (length(x) < 30) {
    return(list(estimate = estimate, ci_low = NA_real_, ci_high = NA_real_, n_valid = length(x)))
  }
  
  alpha <- (1 - conf) / 2
  qs <- stats::quantile(x, probs = c(alpha, 1 - alpha), na.rm = TRUE, names = FALSE, type = 6)
  
  list(
    estimate = estimate,
    ci_low = as.numeric(qs[1]),
    ci_high = as.numeric(qs[2]),
    n_valid = length(x)
  )
}

# Ensures to always have a list with estimate/ci_low/ci_high, even if a function returns atomic
ensure_ci_list <- function(x, fallback_estimate = NA_real_) {
  if (is.list(x)) {
    if (is.null(x[["estimate"]])) x[["estimate"]] <- fallback_estimate
    if (is.null(x[["ci_low"]]))   x[["ci_low"]]   <- NA_real_
    if (is.null(x[["ci_high"]]))  x[["ci_high"]]  <- NA_real_
    if (is.null(x[["n_valid"]]))  x[["n_valid"]]  <- NA_integer_
    return(x)
  }
  # atomic -> wrap
  list(
    estimate = suppressWarnings(as.numeric(x))[1] %||% fallback_estimate,
    ci_low = NA_real_,
    ci_high = NA_real_,
    n_valid = NA_integer_
  )
}

# Bootstrap stats
# Correlation with bootstrap CI (Pearson or Spearman)
corr_boot <- function(df, x, y, method = c("pearson", "spearman"), R = 5000, conf = 0.95) {
  method <- match.arg(method)
  d <- df %>% select(all_of(c(x, y))) %>% drop_na()
  
  est <- suppressWarnings(cor(d[[1]], d[[2]], method = method))
  
  boot_stats <- replicate(R, {
    idx <- sample.int(nrow(d), size = nrow(d), replace = TRUE)
    dd <- d[idx, , drop = FALSE]
    suppressWarnings(cor(dd[[1]], dd[[2]], method = method))
  })
  
  boot_ci(boot_stats, estimate = est, conf = conf)
}

# Cohen's d (independent groups) + stratified bootstrap CI (preregistered estimand)
cohens_d_boot <- function(df, score_col, group_col = "condition", R = 5000, conf = 0.95) {
  d <- df %>% select(all_of(c(score_col, group_col))) %>% drop_na()
  d[[group_col]] <- as_good_bad(d[[group_col]])
  
  x_bad  <- d %>% filter(.data[[group_col]] == "bad")  %>% pull(.data[[score_col]])
  x_good <- d %>% filter(.data[[group_col]] == "good") %>% pull(.data[[score_col]])
  
  if (length(x_bad) < 2 || length(x_good) < 2) {
    return(list(estimate = NA_real_, ci_low = NA_real_, ci_high = NA_real_, n_valid = 0))
  }
  
  d_point <- function(a, b) {
    n1 <- length(a); n2 <- length(b)
    s1 <- stats::sd(a); s2 <- stats::sd(b)
    sp <- sqrt(((n1 - 1) * s1^2 + (n2 - 1) * s2^2) / (n1 + n2 - 2))
    if (!is.finite(sp) || sp == 0) return(NA_real_)
    (mean(b) - mean(a)) / sp  # good - bad (positive -> good > bad)
  }
  
  est <- d_point(x_bad, x_good)
  
  boot_stats <- replicate(R, {
    a <- sample(x_bad,  size = length(x_bad),  replace = TRUE)
    b <- sample(x_good, size = length(x_good), replace = TRUE)
    d_point(a, b)
  })
  
  boot_ci(boot_stats, estimate = est, conf = conf)
}

# ROC AUC + stratified bootstrap CI (preregistered estimand)
auc_boot <- function(df, score_col, group_col = "condition", R = 5000, conf = 0.95) {
  d <- df %>% select(all_of(c(score_col, group_col))) %>% drop_na()
  d[[group_col]] <- as_good_bad(d[[group_col]])
  
  bad_df  <- d %>% filter(.data[[group_col]] == "bad")
  good_df <- d %>% filter(.data[[group_col]] == "good")
  
  if (nrow(bad_df) < 2 || nrow(good_df) < 2) {
    return(list(estimate = NA_real_, ci_low = NA_real_, ci_high = NA_real_, n_valid = 0))
  }
  
  auc_point <- function(dd) {
    if (length(unique(dd[[group_col]])) < 2) return(NA_real_)
    roc_obj <- pROC::roc(
      response = dd[[group_col]],
      predictor = dd[[score_col]],
      levels = c("bad", "good"),
      direction = "<",
      quiet = TRUE
    )
    as.numeric(pROC::auc(roc_obj))
  }
  
  est <- auc_point(d)
  
  boot_stats <- replicate(R, {
    b <- bad_df  %>% slice_sample(n = nrow(bad_df),  replace = TRUE)
    g <- good_df %>% slice_sample(n = nrow(good_df), replace = TRUE)
    dd <- bind_rows(b, g)
    auc_point(dd)
  })
  
  boot_ci(boot_stats, estimate = est, conf = conf)
}

# Bland–Altman (LLM - HumanGold) + bootstrap CI for bias and LoA (preregistered)
bland_altman_boot <- function(df, human_col, llm_col, R = 5000, conf = 0.95) {
  d <- df %>% select(all_of(c(human_col, llm_col))) %>% drop_na()
  diff <- d[[llm_col]] - d[[human_col]]
  
  est_bias <- mean(diff)
  est_loa_low  <- mean(diff) - 1.96 * sd(diff)
  est_loa_high <- mean(diff) + 1.96 * sd(diff)
  
  boot_bias <- replicate(R, {
    idx <- sample.int(length(diff), size = length(diff), replace = TRUE)
    mean(diff[idx])
  })
  
  boot_loa_low <- replicate(R, {
    idx <- sample.int(length(diff), size = length(diff), replace = TRUE)
    x <- diff[idx]
    mean(x) - 1.96 * sd(x)
  })
  
  boot_loa_high <- replicate(R, {
    idx <- sample.int(length(diff), size = length(diff), replace = TRUE)
    x <- diff[idx]
    mean(x) + 1.96 * sd(x)
  })
  
  list(
    bias = boot_ci(boot_bias, estimate = est_bias, conf = conf),
    loa_low = boot_ci(boot_loa_low, estimate = est_loa_low, conf = conf),
    loa_high = boot_ci(boot_loa_high, estimate = est_loa_high, conf = conf),
    sd_diff = sd(diff),
    n = length(diff)
  )
}

# Read & clean
read_clean <- function(path) {
  readr::read_csv(path, show_col_types = FALSE) %>%
    janitor::clean_names()
}

human_primary_gold <- read_clean(PATH_HUMAN_PRIMARY_GOLD)
human_primary_rows <- read_clean(PATH_HUMAN_PRIMARY_ROWS)
human_sens_gold    <- read_clean(PATH_HUMAN_SENS_GOLD)
human_sens_rows    <- read_clean(PATH_HUMAN_SENS_ROWS)

llm_primary <- read_clean(PATH_LLM_PRIMARY_SUBSET)
llm_sens    <- read_clean(PATH_LLM_SENS_SUBSET)
llm_runs    <- read_clean(PATH_LLM_RUNS)

# Standardize column names
llm_runs <- llm_runs %>%
  mutate(
    dialog_id = as.character(dialog_id),
    run_id    = as.integer(run_id)
  )

# Optional:
# llm_all <- read_clean(PATH_LLM_ALL)

# Dataset-specific standardization
SCALES_7 <- c(
  "truthfulness", "relevance", "clarity",
  "relational_appropriateness", "logic_coherence",
  "respect_appreciation", "feedback_depth"
)
SCALE_OVERALL <- "overall"

std_human_gold <- function(df) {
  stop_if_missing(df, c("dialog_id", "overall"), "human_gold")
  
  # Primary mapping
  out <- df %>%
    transmute(
      dialog_id = dialog_id,
      truthfulness = human_gold_truthfulness,
      relevance = human_gold_relevance,
      clarity = human_gold_clarity,
      relational_appropriateness = relation_appropriateness,
      logic_coherence = human_gold_logic_coherence,
      respect_appreciation = human_gold_respect_appreciation,
      feedback_depth = human_gold_feedback_depth,
      overall = overall
    )
  out
}

std_human_rows <- function(df) {
  stop_if_missing(df, c("dialog_id", "rater_id", "condition", "overall"), "human_rows")
  
  out <- df %>%
    transmute(
      dialog_id = dialog_id,
      rater_id = rater_id,
      condition = as_good_bad(condition),
      truthfulness = truthfulness,
      relevance = relevance,
      clarity = clarity,
      relational_appropriateness = relation_appropriateness,
      logic_coherence = logic_coherence,
      respect_appreciation = respect_appreciation,
      feedback_depth = feedback_depth,
      overall = overall
    )
  out
}

std_llm_median <- function(df) {
  stop_if_missing(df, c("dialog_id", "condition", "llm_median_overall_quality"), "llm_median")
  
  out <- df %>%
    transmute(
      dialog_id = dialog_id,
      condition = as_good_bad(condition),
      truthfulness = llm_median_truthfulness,
      relevance = llm_median_relevance,
      clarity = llm_median_clarity,
      relational_appropriateness = llm_median_relational_appropriateness,
      logic_coherence = llm_median_logic_coherence,
      respect_appreciation = llm_median_respect_appreciation,
      feedback_depth = llm_median_feedback_depth,
      overall = llm_median_overall_quality
    )
  out
}

# Standardize
human_primary_gold <- std_human_gold(human_primary_gold)
human_sens_gold    <- std_human_gold(human_sens_gold)

llm_primary <- std_llm_median(llm_primary)
llm_sens    <- std_llm_median(llm_sens)

human_primary_rows <- std_human_rows(human_primary_rows)
human_sens_rows    <- std_human_rows(human_sens_rows)

# Merge for H1/H2/H8
merge_gold_llm <- function(hg, llm) {
  stop_if_missing(hg,  c("dialog_id", "overall"), "human_gold_std")
  stop_if_missing(llm, c("dialog_id", "condition", "overall"), "llm_std")
  
  d <- hg %>%
    rename_with(\(x) paste0("human_", x), all_of(c(SCALES_7, SCALE_OVERALL))) %>%
    inner_join(
      llm %>% rename_with(\(x) paste0("llm_", x), all_of(c(SCALES_7, SCALE_OVERALL))),
      by = "dialog_id"
    ) %>%
    mutate(condition = llm$condition[match(dialog_id, llm$dialog_id)]) %>%
    mutate(condition = as_good_bad(condition))
  
  d
}

dat_primary <- merge_gold_llm(human_primary_gold, llm_primary)
dat_sens    <- merge_gold_llm(human_sens_gold, llm_sens)

# H1/H2: correlations
run_correlations <- function(dat, R = 5000) {
  h1_p <- corr_boot(dat, "llm_overall", "human_overall", method = "pearson", R = R)
  h1_s <- corr_boot(dat, "llm_overall", "human_overall", method = "spearman", R = R)
  
  sub_results <- map_dfr(SCALES_7, function(sc) {
    xp <- corr_boot(dat, paste0("llm_", sc), paste0("human_", sc), method = "pearson", R = R)
    xs <- corr_boot(dat, paste0("llm_", sc), paste0("human_", sc), method = "spearman", R = R)
    
    d2 <- dat %>% select(all_of(c(paste0("llm_", sc), paste0("human_", sc)))) %>% drop_na()
    p_p <- suppressWarnings(cor.test(d2[[1]], d2[[2]], method = "pearson")$p.value)
    p_s <- suppressWarnings(cor.test(d2[[1]], d2[[2]], method = "spearman", exact = FALSE)$p.value)
    
    tibble(
      scale = sc,
      pearson_r = xp$estimate, pearson_ci_low = xp$ci_low, pearson_ci_high = xp$ci_high, pearson_p = p_p,
      spearman_rho = xs$estimate, spearman_ci_low = xs$ci_low, spearman_ci_high = xs$ci_high, spearman_p = p_s
    )
  }) %>%
    mutate(
      pearson_p_fdr  = p.adjust(pearson_p, method = "BH"),
      spearman_p_fdr = p.adjust(spearman_p, method = "BH")
    )
  
  list(
    h1 = tibble(
      measure = c("pearson_r", "spearman_rho"),
      estimate = c(h1_p$estimate, h1_s$estimate),
      ci_low = c(h1_p$ci_low, h1_s$ci_low),
      ci_high = c(h1_p$ci_high, h1_s$ci_high)
    ),
    h2 = sub_results
  )
}

corr_primary <- run_correlations(dat_primary, R = 5000)
corr_sens    <- run_correlations(dat_sens, R = 5000)

# H3: ICC(2,2) absolute agreement
run_icc <- function(rows_df, rater_col = "rater_id") {
  
  map_dfr(c(SCALES_7, SCALE_OVERALL), function(sc) {
    
    long <- rows_df %>%
      select(dialog_id, rater = all_of(rater_col), value = all_of(sc)) %>%
      drop_na() %>%
      distinct()
    
    # For each dialog, keep exactly 2 ratings
    pair <- long %>%
      arrange(dialog_id, rater) %>%
      group_by(dialog_id) %>%
      filter(n() >= 2) %>%
      slice_head(n = 2) %>%
      summarise(
        r1 = first(value),
        r2 = last(value),
        .groups = "drop"
      )
    
    # Need enough dialogs
    if (nrow(pair) < 5) {
      return(tibble(scale = sc, icc = NA_real_, ci_low = NA_real_, ci_high = NA_real_, n_dialogs = nrow(pair)))
    }
    
    # Ensure numeric
    mat <- pair %>%
      transmute(
        r1 = suppressWarnings(as.numeric(as.character(r1))),
        r2 = suppressWarnings(as.numeric(as.character(r2)))
      ) %>%
      as.data.frame()
    
    # Drop rows with missing after coercion
    mat <- mat[complete.cases(mat), , drop = FALSE]
    
    if (nrow(mat) < 5) {
      return(tibble(scale = sc, icc = NA_real_, ci_low = NA_real_, ci_high = NA_real_, n_dialogs = nrow(mat)))
    }
    
    icc_obj <- irr::icc(
      mat,
      model = "twoway",
      type = "agreement",
      unit = "average",
      conf.level = 0.95
    )
    
    tibble(
      scale = sc,
      icc = unname(icc_obj$value),
      ci_low = unname(icc_obj$lbound),
      ci_high = unname(icc_obj$ubound),
      n_dialogs = nrow(mat)
    )
  })
}

icc_primary <- run_icc(human_primary_rows, "rater_id")
icc_sens    <- run_icc(human_sens_rows, "rater_id")

# H4/H5: Discriminability (AUC + Cohen's d)
attach_condition <- function(gold_df, llm_df) {
  gold_df %>%
    left_join(llm_df %>% select(dialog_id, condition), by = "dialog_id") %>%
    mutate(condition = as_good_bad(condition))
}

run_discriminability <- function(df, score_col = "overall", R = 5000, conf = 0.95) {
  df2 <- df %>% select(condition, all_of(score_col)) %>% drop_na()
  df2$condition <- as_good_bad(df2$condition)
  
  auc <- ensure_ci_list(auc_boot(df2, score_col, "condition", R = R, conf = conf))
  d   <- ensure_ci_list(cohens_d_boot(df2, score_col, "condition", R = R, conf = conf))
  
  # Use [[ ]] access only (no $) to avoid atomic-$ crashes
  auc_est <- auc[["estimate"]]; auc_lo <- auc[["ci_low"]]; auc_hi <- auc[["ci_high"]]
  d_est   <- d[["estimate"]];   d_lo   <- d[["ci_low"]];   d_hi   <- d[["ci_high"]]
  
  tibble(
    score = score_col,
    auc = as.numeric(auc_est), auc_ci_low = as.numeric(auc_lo), auc_ci_high = as.numeric(auc_hi),
    cohens_d = as.numeric(d_est), d_ci_low = as.numeric(d_lo), d_ci_high = as.numeric(d_hi)
  )
}

human_primary_gold_with_cond <- attach_condition(human_primary_gold, llm_primary)
human_sens_gold_with_cond    <- attach_condition(human_sens_gold, llm_sens)

disc_primary_human <- run_discriminability(human_primary_gold_with_cond, "overall", R = 5000)
disc_primary_llm   <- run_discriminability(llm_primary, "overall", R = 5000)

disc_sens_human <- run_discriminability(human_sens_gold_with_cond, "overall", R = 5000)
disc_sens_llm   <- run_discriminability(llm_sens, "overall", R = 5000)

# H4/H5: Discriminability per subscale
# Safe wrappers with tryCatch are used because possible AUC = 1.00 (perfect separation)

run_discriminability_subscales <- function(df, scale_cols, R = 5000, conf = 0.95) {
  map_dfr(scale_cols, function(sc) {
    
    df2 <- df %>% select(condition, all_of(sc)) %>% drop_na()
    df2$condition <- as_good_bad(df2$condition)
    
    # Safe AUC wrapper
    auc_result <- tryCatch({
      raw <- auc_boot(df2, sc, "condition", R = R, conf = conf)
      ensure_ci_list(raw)
    }, error = function(e) {
      list(estimate = NA_real_, ci_low = NA_real_, ci_high = NA_real_, n_valid = NA_integer_)
    })
    
    # Safe Cohen's d wrapper
    d_result <- tryCatch({
      raw <- cohens_d_boot(df2, sc, "condition", R = R, conf = conf)
      ensure_ci_list(raw)
    }, error = function(e) {
      list(estimate = NA_real_, ci_low = NA_real_, ci_high = NA_real_, n_valid = NA_integer_)
    })
    
    # Safe extraction with fallback
    safe_num <- function(x, key) {
      val <- tryCatch(x[[key]], error = function(e) NA_real_)
      if (is.null(val) || length(val) == 0) return(NA_real_)
      as.numeric(val)
    }
    
    tibble(
      scale       = sc,
      auc         = safe_num(auc_result, "estimate"),
      auc_ci_low  = safe_num(auc_result, "ci_low"),
      auc_ci_high = safe_num(auc_result, "ci_high"),
      cohens_d    = safe_num(d_result,   "estimate"),
      d_ci_low    = safe_num(d_result,   "ci_low"),
      d_ci_high   = safe_num(d_result,   "ci_high")
    )
  })
}

# PRIMARY - Human subscales
disc_primary_human_sub <- run_discriminability_subscales(
  human_primary_gold_with_cond,
  scale_cols = SCALES_7,
  R = 5000
)

# PRIMARY - LLM subscales
disc_primary_llm_sub <- run_discriminability_subscales(
  llm_primary,
  scale_cols = c(SCALES_7, "overall"),
  R = 5000
)

# SENS - Human subscales
disc_sens_human_sub <- run_discriminability_subscales(
  human_sens_gold_with_cond,
  scale_cols = SCALES_7,
  R = 5000
)

# SENS - LLM subscales
disc_sens_llm_sub <- run_discriminability_subscales(
  llm_sens,
  scale_cols = c(SCALES_7, "overall"),
  R = 5000
)

cat("\n PRIMARY: H5 LLM discriminability (subscales) \n")
print(disc_primary_llm_sub)

cat("\n SENS: H5 LLM discriminability (subscales) \n")
print(disc_sens_llm_sub)

# H8: Bland–Altman (overall)
ba_primary <- bland_altman_boot(dat_primary, "human_overall", "llm_overall", R = 5000)
ba_sens    <- bland_altman_boot(dat_sens, "human_overall", "llm_overall", R = 5000)

ba_to_tbl <- function(ba) {
  # Use [[ ]] access only
  tibble(
    n = ba$n,
    bias = ba$bias[["estimate"]], bias_ci_low = ba$bias[["ci_low"]], bias_ci_high = ba$bias[["ci_high"]],
    loa_low = ba$loa_low[["estimate"]], loa_low_ci_low = ba$loa_low[["ci_low"]], loa_low_ci_high = ba$loa_low[["ci_high"]],
    loa_high = ba$loa_high[["estimate"]], loa_high_ci_low = ba$loa_high[["ci_low"]], loa_high_ci_high = ba$loa_high[["ci_high"]],
    sd_diff = ba$sd_diff
  )
}

ba_primary_tbl <- ba_to_tbl(ba_primary)
ba_sens_tbl    <- ba_to_tbl(ba_sens)

# LLM Intra-Model Reliability (ICC across 3 runs)

# LLM scale keys
LLM_RUN_SCALES <- c(
  "llm_clarity",
  "llm_relevance",
  "llm_truthfulness",
  "llm_logic_coherence",
  "llm_respect_appreciation",
  "llm_relational_appropriateness",
  "llm_feedback_depth",
  "llm_overall_quality"
)

run_llm_icc <- function(runs_df, scale_cols = LLM_RUN_SCALES, n_runs = 3) {
  
  # Only regular dialogs (no Gold or IMC control dialogs)
  runs_df <- runs_df %>%
    filter(
      is.na(is_gold) | is_gold == 0,
      is.na(is_imc)  | is_imc  == 0
    )
  
  map_dfr(scale_cols, function(sc) {
    
    # Keep only dialogs with exactly n_runs complete runs
    wide <- runs_df %>%
      select(dialog_id, run_id, value = all_of(sc)) %>%
      drop_na() %>%
      distinct(dialog_id, run_id, .keep_all = TRUE) %>%
      pivot_wider(
        names_from  = run_id,
        values_from = value,
        names_prefix = "run_"
      ) %>%
      filter(if_all(starts_with("run_"), ~ !is.na(.)))
    
    run_cols <- paste0("run_", 1:n_runs)
    missing_cols <- setdiff(run_cols, names(wide))
    if (length(missing_cols) > 0) {
      return(tibble(
        scale     = sc,
        icc_21    = NA_real_, icc_21_ci_low = NA_real_, icc_21_ci_high = NA_real_,
        icc_23    = NA_real_, icc_23_ci_low = NA_real_, icc_23_ci_high = NA_real_,
        n_dialogs = 0L,
        note      = paste("Missing run columns:", paste(missing_cols, collapse = ", "))
      ))
    }
    
    mat <- wide %>%
      select(all_of(run_cols)) %>%
      mutate(across(everything(), ~ suppressWarnings(as.numeric(.)))) %>%
      as.data.frame() %>%
      filter(complete.cases(.))
    
    if (nrow(mat) < 5) {
      return(tibble(
        scale     = sc,
        icc_21    = NA_real_, icc_21_ci_low = NA_real_, icc_21_ci_high = NA_real_,
        icc_23    = NA_real_, icc_23_ci_low = NA_real_, icc_23_ci_high = NA_real_,
        n_dialogs = nrow(mat),
        note      = "Too few complete cases"
      ))
    }
    
    # ICC(2,1): two-way random, absolute agreement, single run
    # Conservative estimate: reliability of ONE single run
    # Directly comparable to a single human rater
    icc_21_obj <- irr::icc(
      mat,
      model      = "twoway",
      type       = "agreement",
      unit       = "single",      # <-- single run
      conf.level = 0.95
    )
    
    # ICC(2,3): two-way random, absolute agreement, average of 3 runs
    # Reflects reliability of the aggregated median score (as actually used)
    # Structurally higher than ICC(2,1) due to Spearman-Brown
    icc_23_obj <- irr::icc(
      mat,
      model      = "twoway",
      type       = "agreement",
      unit       = "average",     # <-- average of 3 runs
      conf.level = 0.95
    )
    
    tibble(
      scale          = sc,
      icc_21         = round(unname(icc_21_obj$value),  3),
      icc_21_ci_low  = round(unname(icc_21_obj$lbound), 3),
      icc_21_ci_high = round(unname(icc_21_obj$ubound), 3),
      icc_23         = round(unname(icc_23_obj$value),  3),
      icc_23_ci_low  = round(unname(icc_23_obj$lbound), 3),
      icc_23_ci_high = round(unname(icc_23_obj$ubound), 3),
      n_dialogs      = nrow(mat),
      note           = ""
    )
  })
}

llm_icc_intra <- run_llm_icc(llm_runs)

# Save
write_csv(llm_icc_intra, "outputs/LLM_intra_model_ICC.csv")

cat("\n LLM Intra-Model Reliability \n")
cat("ICC(2,1) = reliability of a single run (conservative, comparable to human ICC)\n")
cat("ICC(2,3) = reliability of aggregated median score (as actually used in analyses)\n\n")
print(llm_icc_intra)

# Summary: statistics across all scales
cat("\n Summary across all scales \n")
llm_icc_intra %>%
  filter(is.na(note) | note == "") %>%
  summarise(
    icc_21_mean = round(mean(icc_21, na.rm = TRUE), 3),
    icc_21_min  = round(min(icc_21,  na.rm = TRUE), 3),
    icc_21_max  = round(max(icc_21,  na.rm = TRUE), 3),
    icc_23_mean = round(mean(icc_23, na.rm = TRUE), 3),
    icc_23_min  = round(min(icc_23,  na.rm = TRUE), 3),
    icc_23_max  = round(max(icc_23,  na.rm = TRUE), 3)
  ) %>%
  print()


# H9: Robustness checks (optional)
run_h9_optional <- function(dat) {
  cov_candidates <- c("dialog_length", "length", "n_tokens", "readability", "fkgl", "flesch", "turn_count")
  covs <- intersect(cov_candidates, names(dat))
  if (length(covs) == 0) {
    return(tibble(note = "No preregistered covariates present in dataset; H9 robustness not run."))
  }
  
  d <- dat %>% select(llm_overall, human_overall, all_of(covs)) %>% drop_na()
  if (nrow(d) < 20) return(tibble(note = "Too few complete cases for robustness."))
  
  res_llm   <- resid(lm(llm_overall ~ ., data = d %>% select(-human_overall)))
  res_human <- resid(lm(human_overall ~ ., data = d %>% select(-llm_overall)))
  r_partial <- cor(res_llm, res_human, method = "pearson")
  
  tibble(
    covariates = paste(covs, collapse = ", "),
    partial_r = r_partial,
    n = nrow(d)
  )
}

h9_primary <- run_h9_optional(dat_primary)
h9_sens    <- run_h9_optional(dat_sens)

# Compile outputs
results_primary <- list(
  H1 = corr_primary$h1,
  H2 = corr_primary$h2,
  H3_ICC = icc_primary,
  H4_human = disc_primary_human,
  H4_human_sub = disc_primary_human_sub,
  H5_llm = disc_primary_llm,
  H5_llm_sub = disc_primary_llm_sub,
  H8_bland_altman = ba_primary_tbl,
  H9_optional = h9_primary,
  LLM_intra_ICC = llm_icc_intra
)

results_sens <- list(
  H1 = corr_sens$h1,
  H2 = corr_sens$h2,
  H3_ICC = icc_sens,
  H4_human = disc_sens_human,
  H4_human_sub = disc_sens_human_sub,
  H5_llm = disc_sens_llm,
  H5_llm_sub = disc_sens_llm_sub,
  H8_bland_altman = ba_sens_tbl,
  H9_optional = h9_sens,
  LLM_intra_ICC = llm_icc_intra
)

# Save as CSV
dir.create("outputs", showWarnings = FALSE)

write_csv(results_primary$H1, "outputs/H1_primary.csv")
write_csv(results_primary$H2, "outputs/H2_primary.csv")
write_csv(results_primary$H3_ICC, "outputs/H3_ICC_primary.csv")
write_csv(results_primary$H4_human, "outputs/H4_primary_human.csv")
write_csv(results_primary$H4_human_sub, "outputs/H4_primary_human_subscales.csv")
write_csv(results_primary$H5_llm, "outputs/H5_primary_llm.csv")
write_csv(results_primary$H5_llm_sub, "outputs/H5_primary_llm_subscales.csv")
write_csv(results_primary$H8_bland_altman, "outputs/H8_primary_bland_altman.csv")
write_csv(results_primary$H9_optional, "outputs/H9_primary_optional.csv")
write_csv(llm_icc_intra, "outputs/LLM_intra_model_ICC.csv")

write_csv(results_sens$H1, "outputs/H1_sens.csv")
write_csv(results_sens$H2, "outputs/H2_sens.csv")
write_csv(results_sens$H3_ICC, "outputs/H3_ICC_sens.csv")
write_csv(results_sens$H4_human, "outputs/H4_sens_human.csv")
write_csv(results_sens$H4_human_sub, "outputs/H4_sens_human_subscales.csv") 
write_csv(results_sens$H5_llm, "outputs/H5_sens_llm.csv")
write_csv(results_sens$H5_llm_sub, "outputs/H5_sens_llm_subscales.csv") 
write_csv(results_sens$H8_bland_altman, "outputs/H8_sens_bland_altman.csv")
write_csv(results_sens$H9_optional, "outputs/H9_sens_optional.csv")

# Quick console summary
cat("\n PRIMARY: H1 Overall correlations \n")
print(results_primary$H1)

cat("\n PRIMARY: H2 Subscales (with BH/FDR p-values) \n")
print(results_primary$H2)

cat("\n PRIMARY: H3 ICC(2,2) \n")
print(results_primary$H3_ICC)

cat("\n PRIMARY: H4 Human discriminability (overall) \n")
print(results_primary$H4_human)

cat("\n PRIMARY: H4 Human discriminability (subscales) \n")
print(results_primary$H4_human_sub)

cat("\n PRIMARY: H5 LLM discriminability (overall) \n")
print(results_primary$H5_llm)

cat("\n PRIMARY: H5 LLM discriminability (subscales) \n")
print(results_primary$H5_llm_sub)

cat("\n PRIMARY: H8 Bland–Altman (overall) \n")
print(results_primary$H8_bland_altman)

cat("\n PRIMARY: H9 Optional robustness \n")
print(results_primary$H9_optional)


cat(" SENS: H1 Overall correlations \n")
print(results_sens$H1)

cat("\n SENS: H2 Subscales (with BH/FDR p-values) \n")
print(results_sens$H2)

cat("\n SENS: H3 ICC(2,2) \n")
print(results_sens$H3_ICC)

cat("\n SENS: H4 Human discriminability (overall) \n")
print(results_sens$H4_human)

cat("\n SENS: H4 Human discriminability (subscales) \n")
print(results_sens$H4_human_sub)

cat("\n SENS: H5 LLM discriminability (overall) \n")
print(results_sens$H5_llm)

cat("\n SENS: H5 LLM discriminability (subscales) \n")
print(results_sens$H5_llm_sub)

cat("\n SENS: H8 Bland–Altman (overall) \n")
print(results_sens$H8_bland_altman)

cat("\n SENS: H9 Optional robustness \n")
print(results_sens$H9_optional)

cat("\n LLM Intra-Model Reliability \n")
cat("ICC(2,1) = reliability of a single run (conservative)\n")
cat("ICC(2,3) = reliability of aggregated median score (as used)\n\n")
print(llm_icc_intra)

library(tidyverse)

corr_data <- read_csv("outputs/H2_primary.csv", show_col_types = FALSE)

heatmap_data <- corr_data %>%
  select(scale, pearson_r) %>%
  mutate(scale = factor(scale, levels = scale))

ggplot(heatmap_data, aes(x = 1, y = scale, fill = pearson_r)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(pearson_r, 2)), color = "black", size = 5) +
  scale_fill_gradient2(low = "#d73027", mid = "white", high = "#1a9850",
                       midpoint = 0.5, limits = c(0,1),
                       name = "Pearson r") +
  labs(
    title = "Convergent Validity: LLM vs Human Gold Standard",
    x = "",
    y = ""
  ) +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_blank(),
        axis.ticks = element_blank())

dir.create("figures", showWarnings = FALSE)

heatmap_plot <- ggplot(heatmap_data, aes(x = 1, y = scale, fill = pearson_r)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(pearson_r, 2)), size = 5) +
  scale_fill_gradient2(low = "#d73027", mid = "white", high = "#1a9850",
                       midpoint = 0.5, limits = c(0,1),
                       name = "Pearson r") +
  labs(
    title = "Convergent Validity: LLM vs Human Gold Standard",
    x = "",
    y = ""
  ) +
  theme_minimal(base_size = 14) +
  theme(axis.text.x = element_blank(),
        axis.ticks = element_blank())

ggsave(
  filename = "figures/Figure_StudyA_Heatmap_Correlations.png",
  plot = heatmap_plot,
  width = 6,
  height = 5,
  dpi = 300
)


icc_data <- read_csv("outputs/H3_ICC_primary.csv", show_col_types = FALSE)

ggplot(icc_data, aes(x = icc, y = reorder(scale, icc))) +
  geom_errorbar(
    aes(xmin = ci_low, xmax = ci_high),
    orientation = "y",
    height = 0.25,
    linewidth = 0.8,
    color = "grey40"
  ) +
  geom_point(size = 3.5, color = "black") +
  geom_vline(xintercept = 0.70, linetype = "dashed", linewidth = 0.7) +
  geom_vline(xintercept = 0.90, linetype = "dotted", linewidth = 0.7) +
  scale_x_continuous(limits = c(0,1)) +
  labs(
    title = "Interrater Reliability of the Human Gold Standard",
    subtitle = "ICC(2,2), Absolute Agreement",
    x = "Intraclass Correlation Coefficient",
    y = NULL
  ) +
  theme_minimal(base_size = 14)

forest_plot <- ggplot(icc_data, aes(x = icc, y = reorder(scale, icc))) +
  geom_errorbar(
    aes(xmin = ci_low, xmax = ci_high),
    orientation = "y",
    height = 0.25
  ) +
  geom_point(size = 3) +
  geom_vline(xintercept = 0.70, linetype = "dashed") +
  geom_vline(xintercept = 0.90, linetype = "dotted") +
  scale_x_continuous(limits = c(0,1)) +
  labs(
    title = "Interrater Reliability of the Human Gold Standard",
    subtitle = "ICC(2,2), Absolute Agreement",
    x = "Intraclass Correlation Coefficient",
    y = NULL
  ) +
  theme_minimal(base_size = 14)

ggsave(
  filename = "figures/Figure_StudyA_ICC_Forest.png",
  plot = forest_plot,
  width = 7,
  height = 5,
  dpi = 300
)
