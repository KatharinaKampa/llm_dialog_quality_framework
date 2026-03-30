# STUDY B – Statistical Analysis Script (PRIMARY + SENS)

# Preregistration: OSF (Preregistration_StudyB.pdf)
# Data:            studyB_all_ratings_cleaned.csv      (PRIMARY, N=139)
#                  studyB_all_ratings_cleaned_SENS.csv  (SENS, N=153)
#                  llm_median_all_dialogs.csv           (LLM-Submetriken Study A)
# R Version:       >= 4.2.0

# -----
# PACKAGES
required_packages <- c(
  "tidyverse", "car", "emmeans", "effectsize", "mediation",
  "lme4", "performance", "parameters", "boot", "rstatix",
  "ggpubr", "knitr", "patchwork", "sandwich", "lmtest"
)
new_pkgs <- required_packages[!required_packages %in% installed.packages()[,"Package"]]
if (length(new_pkgs)) install.packages(new_pkgs, dependencies = FALSE, type = "binary")
invisible(lapply(required_packages, library, character.only = TRUE))

# -----
# Load & Merge data
# Both datasets (PRIMARY + SENS) are loaded and merged with LLM

# Paths for all datasets
PATH_PRIMARY <- "studyB_all_ratings_cleaned.csv"
PATH_SENS    <- "studyB_all_ratings_cleaned_SENS.csv"
PATH_LLM     <- "llm_median_all_dialogs.csv"

llm_cols_to_merge <- c(
  "dialog_id",
  "llm_median_clarity", "llm_median_relevance", "llm_median_truthfulness",
  "llm_median_logic_coherence", "llm_median_respect_appreciation",
  "llm_median_relational_appropriateness", "llm_median_feedback_depth",
  "llm_median_overall_quality"
)

# Loading function used for both datasets
load_and_merge <- function(path_clean, path_llm, label) {
  df_raw <- read_csv(path_clean, show_col_types = FALSE)
  df_llm <- read_csv(path_llm,   show_col_types = FALSE)
  df <- df_raw %>%
    left_join(df_llm %>% select(all_of(llm_cols_to_merge)), by = "dialog_id")
  n_missing_llm <- sum(is.na(df$llm_median_overall_quality))
  cat(sprintf("\n[%s] N = %d | LLM missing matches = %d\n",
              label, nrow(df), n_missing_llm))
  df
}

# Load both datasets
df_primary_raw <- load_and_merge(PATH_PRIMARY, PATH_LLM, "PRIMARY")
df_sens_raw    <- load_and_merge(PATH_SENS,    PATH_LLM, "SENS")

# -----
# Pre-Processing
# As a function, so that it can be used for both data sets

preprocess <- function(df) {
  df %>%
    mutate(
      condition    = factor(condition, levels = c("bad", "good")),
      recipe_title = as.factor(recipe_title),
      dialog_id    = as.factor(dialog_id),
      
      straightline_flag = {
        post_items <- cbind(
          post_diet_suitability, post_recipe_stars, post_cook_intent, post_save_intent,
          dq_clarity, dq_relevance, dq_respect, dq_logic, dq_coherence, manip_check
        )
        apply(post_items, 1, function(x) {
          x_nonmiss <- x[!is.na(x)]
          if (length(x_nonmiss) < 3) return(0L)
          as.integer(max(table(x_nonmiss)) / length(x_nonmiss) >= 0.80)
        })
      },
      
      delta_diet  = post_diet_suitability - pre_diet_suitability,
      delta_stars = post_recipe_stars     - pre_recipe_stars,
      delta_cook  = post_cook_intent      - pre_cook_intent,
      delta_save  = post_save_intent      - pre_save_intent,
      
      abs_err_fat_pre  = abs(pre_est_fat_g  - true_fat_g),
      abs_err_carb_pre = abs(pre_est_carb_g - true_carb_g),
      abs_err_kcal_pre = abs(pre_est_kcal   - true_kcal),
      
      # LLM Quality Index: Average of the six communicative process metrics
      # Excluded:
      # - llm_median_truthfulness: measures factual alignment with DGE guidelines 
      # (content accuracy), not communicative process quality
      # - llm_median_overall_quality: aggregated overall assessment that already summarises the remaining
      #   dimensions: inclusion would result in double weighting
      llm_quality_index = rowMeans(
        cbind(llm_median_clarity,
              llm_median_relevance,
              llm_median_logic_coherence,
              llm_median_respect_appreciation,
              llm_median_relational_appropriateness,
              llm_median_feedback_depth),
        na.rm = TRUE
      )
    )
}

# Preprocessing for both Datasets
df_primary <- preprocess(df_primary_raw)
df_sens    <- preprocess(df_sens_raw)

# -----
# Helper Functions

run_ancova <- function(data, post_var, pre_var, delta_var = NULL, label = "") {
  cat(sprintf("\n\n%s\n========== ANCOVA: %s ==========\n%s\n",
              strrep("=", 60), label, strrep("=", 60)))
  formula_ancova <- as.formula(
    sprintf("%s ~ condition + %s + recipe_title", post_var, pre_var)
  )
  model <- lm(formula_ancova, data = data)
  cat("\nModel summary:\n"); print(summary(model))
  cat("\nType III ANOVA table:\n"); print(car::Anova(model, type = "III"))
  cat("\nPartial eta²:\n"); print(effectsize::eta_squared(model, partial = TRUE, ci = 0.95))
  emm <- emmeans::emmeans(model, ~ condition)
  cat("\nEstimated marginal means:\n"); print(emm)
  cat("\nPairwise contrast:\n"); print(emmeans::contrast(emm, method = "pairwise"))
  cat("\nCohen's d:\n")
  tryCatch(
    print(effectsize::cohens_d(
      as.formula(sprintf("%s ~ condition", post_var)),
      data = data, pooled_sd = TRUE)),
    error = function(e) cat("  Cohen's d could not be computed:", conditionMessage(e), "\n")
  )
  set.seed(42)
  boot_fn <- function(d, i) {
    m_boot <- lm(formula_ancova, data = d[i, ])
    tryCatch(coef(m_boot)["conditiongood"], error = function(e) NA_real_)
  }
  boot_res <- boot::boot(data = data, statistic = boot_fn, R = 5000)
  cat("\n95% BCa Bootstrap CI (B = 5000):\n")
  tryCatch(print(boot::boot.ci(boot_res, type = "bca")),
           error = function(e) { cat("BCa failed, percentile CI:\n")
             print(boot::boot.ci(boot_res, type = "perc")) })
  if (!is.null(delta_var) && delta_var %in% names(data)) {
    cat(sprintf("\nRobustness t-test on %s:\n", delta_var))
    print(t.test(as.formula(sprintf("%s ~ condition", delta_var)), data = data))
  }
  cat("\nICC check (recipe-level clustering):\n")
  null_model <- lme4::lmer(
    as.formula(sprintf("%s ~ 1 + (1 | recipe_title)", post_var)),
    data = data, REML = TRUE,
    control = lme4::lmerControl(optimizer = "bobyqa")
  )
  icc_num <- tryCatch({
    as.numeric(performance::icc(null_model)$ICC_adjusted)
  }, error = function(e) NA_real_, warning = function(w) NA_real_)
  if (is.na(icc_num)) {
    cat("  ICC = 0 (singular fit) — standard SEs sufficient\n")
  } else if (icc_num > 0.05) {
    cat(sprintf("  ICC = %.3f > .05 — cluster-robust SEs:\n", icc_num))
    print(lmtest::coeftest(model, vcov = sandwich::vcovCL(model, cluster = ~recipe_title)))
  } else {
    cat(sprintf("  ICC = %.3f <= .05 — standard SEs sufficient\n", icc_num))
  }
  invisible(model)
}

check_assumptions <- function(model, pre_var, label) {
  cat(sprintf("\n-- Assumption check: %s --\n", label))
  res <- residuals(model)
  sw <- shapiro.test(res)
  cat(sprintf("  Shapiro-Wilk: W = %.4f, p = %.4f", sw$statistic, sw$p.value))
  if (sw$p.value < 0.01) {
    cat("  <- p < .01: Bootstrap CIs reported\n")
  } else {
    cat("  <- normality assumption met\n")
  }
  int_formula <- update(formula(model),
                        as.formula(sprintf(". ~ . + condition:%s", pre_var)))
  int_model  <- lm(int_formula, data = model$model)
  aov_slopes <- anova(model, int_model)
  p_slopes   <- aov_slopes$`Pr(>F)`[2]
  cat(sprintf("  Homogeneity of slopes (interaction p = %.4f)", p_slopes))
  if (!is.na(p_slopes) && p_slopes < 0.05) {
    cat("  <- VIOLATION: delta-score t-test is the fallback\n")
  } else {
    cat("  <- assumption met\n")
  }
}

extract_ancova_row <- function(model, hypothesis, outcome) {
  cf   <- summary(model)$coefficients
  eta2 <- effectsize::eta_squared(model, partial = TRUE)
  cond_row <- grep("conditiongood", rownames(cf))
  eta_row  <- grep("condition",     eta2$Parameter)
  tibble(
    Hypothesis   = hypothesis,
    Outcome      = outcome,
    B_condition  = ifelse(length(cond_row) > 0, round(cf[cond_row, "Estimate"],   3), NA),
    SE           = ifelse(length(cond_row) > 0, round(cf[cond_row, "Std. Error"], 3), NA),
    p_raw        = ifelse(length(cond_row) > 0, round(cf[cond_row, "Pr(>|t|)"],  4), NA),
    partial_eta2 = ifelse(length(eta_row)  > 0, round(eta2$Eta2_partial[eta_row], 3), NA)
  )
}

extract_p <- function(model, term = "conditiongood") {
  cf <- summary(model)$coefficients
  if (term %in% rownames(cf)) cf[term, "Pr(>|t|)"] else NA_real_
}


# -----
# Main analysis function
# All analyses in a single function called for PRIMARY and SENS

run_all_analyses <- function(df, dataset_label, output_dir = ".") {
  
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Write all output to both the console and the log file simultaneously
  log_path <- file.path(output_dir, sprintf("results_%s.txt", dataset_label))
  sink(log_path, split = TRUE)
  
  cat(sprintf("\n%s\n  STUDY B — DATASET: %s  |  N = %d (good: %d, bad: %d)\n%s\n\n",
              strrep("=", 70), toupper(dataset_label), nrow(df),
              sum(df$condition == "good"), sum(df$condition == "bad"),
              strrep("=", 70)))
  
  # --- QA Verification --------------------------------------------------------
  cat("=== QA Verification ===\n")
  cat(sprintf("  IMC failures:         %d\n", sum(df$imc_pass == 0,              na.rm = TRUE)))
  cat(sprintf("  Time too short:       %d\n", sum(df$min_time_ok_dialog == 0,    na.rm = TRUE)))
  cat(sprintf("  Nutrient outliers:    %d\n", sum(df$nutrient_outlier_flag == 1, na.rm = TRUE)))
  cat(sprintf("  Straightliners:       %d\n", sum(df$straightline_flag == 1,     na.rm = TRUE)))
  cat(sprintf("  Missing post_diet:    %d\n", sum(is.na(df$post_diet_suitability))))
  
  # --- H1 --------------------------------------------------------------------
  cat("\n\n### H1: DIET SUITABILITY ###\n")
  m_h1 <- run_ancova(df, "post_diet_suitability", "pre_diet_suitability",
                     "delta_diet", "H1: Diet Suitability")
  check_assumptions(m_h1, "pre_diet_suitability", "H1")
  
  # --- H2a -------------------------------------------------------------------
  cat("\n\n### H2a: STAR RATING ###\n")
  m_h2a <- run_ancova(df, "post_recipe_stars", "pre_recipe_stars",
                      "delta_stars", "H2a: Star Rating")
  check_assumptions(m_h2a, "pre_recipe_stars", "H2a")
  
  # --- H2b -------------------------------------------------------------------
  cat("\n\n### H2b: COOKING INTENTION ###\n")
  m_h2b <- run_ancova(df, "post_cook_intent", "pre_cook_intent",
                      "delta_cook", "H2b: Cooking Intention")
  check_assumptions(m_h2b, "pre_cook_intent", "H2b")
  
  # --- Explorativ: Saving Intention ------------------------------------------
  cat("\n\n### EXPLORATIV: SAVING INTENTION (descriptive, not confirmatory) ###\n")
  saving_desc <- df %>%
    group_by(condition) %>%
    summarise(
      M_delta_save  = round(mean(delta_save, na.rm = TRUE), 2),
      SD_delta_save = round(sd(delta_save,   na.rm = TRUE), 2),
      n             = sum(!is.na(delta_save)),
      .groups = "drop"
    )
  print(saving_desc)
  cat(sprintf(
    "\n  Δ between conditions (good – bad): %.2f\n",
    saving_desc$M_delta_save[saving_desc$condition == "good"] -
      saving_desc$M_delta_save[saving_desc$condition == "bad"]
  ))
  
  # --- H3 --------------------------------------------------------------------
  cat("\n\n### H3: NUTRIENT ESTIMATION ###\n")
  df_h3_fat  <- df %>% filter(!is.na(abs_err_fat_post),  !is.na(abs_err_fat_pre))
  df_h3_carb <- df %>% filter(!is.na(abs_err_carb_post), !is.na(abs_err_carb_pre))
  df_h3_kcal <- df %>% filter(!is.na(abs_err_kcal_post), !is.na(abs_err_kcal_pre))
  m_h3_fat   <- run_ancova(df_h3_fat,  "abs_err_fat_post",  "abs_err_fat_pre",
                           label = "H3: Fat Error")
  m_h3_carb  <- run_ancova(df_h3_carb, "abs_err_carb_post", "abs_err_carb_pre",
                           label = "H3: Carb Error")
  m_h3_kcal  <- run_ancova(df_h3_kcal, "abs_err_kcal_post", "abs_err_kcal_pre",
                           label = "H3: kcal Error")
  check_assumptions(m_h3_fat,  "abs_err_fat_pre",  "H3 Fat")
  check_assumptions(m_h3_carb, "abs_err_carb_pre", "H3 Carb")
  check_assumptions(m_h3_kcal, "abs_err_kcal_pre", "H3 kcal")
  
  # --- H4 --------------------------------------------------------------------
  cat("\n\n### H4: MANIPULATION CHECK ###\n")
  t_h4 <- t.test(dq_mean ~ condition, data = df, var.equal = FALSE)
  d_h4 <- effectsize::cohens_d(dq_mean ~ condition, data = df)
  print(t_h4); print(d_h4)
  m_h4 <- lm(dq_mean ~ condition + recipe_title, data = df)
  print(car::Anova(m_h4, type = "III"))
  print(emmeans::contrast(emmeans::emmeans(m_h4, ~condition), method = "pairwise"))
  cat("\nSingle-item manipulation check:\n")
  print(t.test(manip_check ~ condition, data = df, var.equal = FALSE))
  print(effectsize::cohens_d(manip_check ~ condition, data = df))
  cat("\nDialog quality sub-scales:\n")
  for (sub in c("dq_clarity", "dq_relevance", "dq_respect", "dq_logic", "dq_coherence")) {
    t_s <- t.test(as.formula(sprintf("%s ~ condition", sub)), data = df, var.equal = FALSE)
    d_s <- effectsize::cohens_d(as.formula(sprintf("%s ~ condition", sub)), data = df)
    cat(sprintf("  %-14s M_good=%.2f  M_bad=%.2f  t(%4.1f)=%6.3f  p=%6.4f  d=%6.3f\n",
                sub,
                mean(df[[sub]][df$condition == "good"], na.rm = TRUE),
                mean(df[[sub]][df$condition == "bad"],  na.rm = TRUE),
                t_s$parameter, t_s$statistic, t_s$p.value, d_s$Cohens_d))
  }
  
  # --- H5 --------------------------------------------------------------------
  cat("\n\n### H5: MEDIATION ###\n")
  cat("NOTE: Preregistered specification uses delta-outcomes (post - pre).\n")
  cat("Corrected to match preregistration: delta_diet and delta_cook as AVs.\n")
  set.seed(42)
  df_med <- droplevels(df)
  med_model <- lm(dq_mean ~ condition, data = df_med)
  print(summary(med_model))
  
  # H5a: delta_diet (accordance with pre-registration)
  out_diet <- lm(delta_diet ~ condition + dq_mean, data = df_med)
  med_diet <- mediation::mediate(med_model, out_diet,
                                 treat = "condition", mediator = "dq_mean",
                                 treat.value = "good", control.value = "bad",
                                 boot = TRUE, sims = 5000, boot.ci.type = "bca")
  cat("\nH5a — Diet Suitability (delta_diet, preregistered):\n")
  print(summary(med_diet))
  
  # H5b: delta_cook (accordance with pre-registration)
  out_cook <- lm(delta_cook ~ condition + dq_mean, data = df_med)
  med_cook <- mediation::mediate(med_model, out_cook,
                                 treat = "condition", mediator = "dq_mean",
                                 treat.value = "good", control.value = "bad",
                                 boot = TRUE, sims = 5000, boot.ci.type = "bca")
  cat("\nH5b — Cooking Intention (delta_cook, preregistered):\n")
  print(summary(med_cook))
  
  # H5c: delta_stars (accordance with pre-registration)
  out_stars <- lm(delta_stars ~ condition + dq_mean, data = df_med)
  med_stars <- mediation::mediate(med_model, out_stars,
                                  treat = "condition", mediator = "dq_mean",
                                  treat.value = "good", control.value = "bad",
                                  boot = TRUE, sims = 5000, boot.ci.type = "bca")
  cat("\nH5c — Star Rating (delta_stars, preregistered):\n")
  print(summary(med_stars))
  
  # --- H6 --------------------------------------------------------------------
  cat("\n\n### H6: LLM QUALITY INDEX -> OPINION CHANGE ###\n")
  m_h6_diet  <- lm(delta_diet  ~ llm_quality_index + pre_diet_suitability +
                     recipe_title + involvement, data = df)
  m_h6_stars <- lm(delta_stars ~ llm_quality_index + pre_recipe_stars +
                     recipe_title + involvement, data = df)
  m_h6_cook  <- lm(delta_cook  ~ llm_quality_index + pre_cook_intent +
                     recipe_title + involvement, data = df)
  cat("\nH6a:\n"); print(summary(m_h6_diet));  print(car::Anova(m_h6_diet,  type = "III"))
  cat("\nH6b:\n"); print(summary(m_h6_stars)); print(car::Anova(m_h6_stars, type = "III"))
  cat("\nH6c:\n"); print(summary(m_h6_cook));  print(car::Anova(m_h6_cook,  type = "III"))
  cat("\nIndividual submetrics (exploratory, delta_diet):\n")
  for (sub in c("llm_median_clarity", "llm_median_relevance", "llm_median_logic_coherence",
                "llm_median_respect_appreciation", "llm_median_relational_appropriateness",
                "llm_median_feedback_depth")) {
    m_sub <- lm(as.formula(sprintf(
      "delta_diet ~ %s + pre_diet_suitability + recipe_title + involvement", sub)),
      data = df)
    cf <- summary(m_sub)$coefficients
    if (sub %in% rownames(cf))
      cat(sprintf("  %-42s  beta=%6.3f  p=%6.4f\n",
                  sub, cf[sub, "Estimate"], cf[sub, "Pr(>|t|)"]))
  }
  
  # --- FDR -------------------------------------------------------------------
  cat("\n\n### FDR CORRECTION (Benjamini-Hochberg) ###\n")
  p_family1 <- c(H2a_stars     = extract_p(m_h2a),
                 H2b_cook      = extract_p(m_h2b),
                 H3_fat_error  = extract_p(m_h3_fat),
                 H3_carb_error = extract_p(m_h3_carb))
  fdr1 <- data.frame(test = names(p_family1),
                     p_raw = round(p_family1, 5),
                     p_fdr = round(p.adjust(p_family1, "BH"), 5))
  cat("\nFamily 1 (H2 + H3):\n"); print(fdr1)
  
  p_family2 <- c(
    H6a = summary(m_h6_diet) $coefficients["llm_quality_index", "Pr(>|t|)"],
    H6b = summary(m_h6_stars)$coefficients["llm_quality_index", "Pr(>|t|)"],
    H6c = summary(m_h6_cook) $coefficients["llm_quality_index", "Pr(>|t|)"]
  )
  fdr2 <- data.frame(test = names(p_family2),
                     p_raw = round(p_family2, 5),
                     p_fdr = round(p.adjust(p_family2, "BH"), 5))
  cat("\nFamily 2 (H6):\n"); print(fdr2)
  
  # --- Results summary table -------------------------------------------------
  cat("\n\n### RESULTS SUMMARY TABLE ###\n")
  results_summary <- bind_rows(
    extract_ancova_row(m_h1,      "H1",  "Diet Suitability"),
    extract_ancova_row(m_h2a,     "H2a", "Star Rating"),
    extract_ancova_row(m_h2b,     "H2b", "Cooking Intention"),
    extract_ancova_row(m_h3_fat,  "H3",  "Abs Error Fat"),
    extract_ancova_row(m_h3_carb, "H3",  "Abs Error Carb"),
    extract_ancova_row(m_h3_kcal, "H3",  "Abs Error kcal")
  )
  results_summary$p_fdr <- c(NA,          # H1:  no FDR (primary Outcome)
                             fdr1$p_fdr, # H2a, H2b, H3-Fat, H3-Carb
                             NA)         # H3-kcal: exploratory, no FDR
  print(as.data.frame(results_summary), row.names = FALSE)
  
  # Save results table as CSV
  write_csv(results_summary,
            file.path(output_dir, sprintf("results_summary_%s.csv", dataset_label)))
  
  cat(sprintf("\n=== ANALYSIS COMPLETE: %s | N = %d ===\n",
              dataset_label, nrow(df)))
  
  # close sink()
  sink()
  cat(sprintf("[%s] Results saved in: %s\n", dataset_label, output_dir))
  
  # Return all models for the PRIMARY vs. SENS comparison
  invisible(list(
    m_h1 = m_h1, m_h2a = m_h2a, m_h2b = m_h2b,
    m_h3_fat = m_h3_fat, m_h3_carb = m_h3_carb, m_h3_kcal = m_h3_kcal,
    m_h4 = m_h4, t_h4 = t_h4, d_h4 = d_h4,
    med_diet = med_diet, med_cook = med_cook,
    m_h6_diet = m_h6_diet, m_h6_stars = m_h6_stars, m_h6_cook = m_h6_cook,
    fdr1 = fdr1, fdr2 = fdr2,
    results_summary = results_summary,
    n = nrow(df)
  ))
}

# -----
# Run analysis
# One time for PRIMARY, one time for SENS

results_primary <- run_all_analyses(df_primary, "PRIMARY", output_dir = "output_PRIMARY")
results_sens    <- run_all_analyses(df_sens,    "SENS",    output_dir = "output_SENS")


# -----
# PRIMARY vs. SENS Comparison (for Chapter 10, master's thesis)
# Direct comparison of key figures from both datasets

compare_primary_sens <- function(res_primary, res_sens, df_primary, df_sens) {
  
  extract_row <- function(res, df, label) {
    tibble(
      Dataset  = label,
      N        = res$n,
      H1_B     = coef(res$m_h1)["conditiongood"],
      H1_p     = summary(res$m_h1)$coefficients["conditiongood", "Pr(>|t|)"],
      H1_eta2  = effectsize::eta_squared(res$m_h1, partial = TRUE)$Eta2_partial[
        grep("condition", effectsize::eta_squared(res$m_h1,
                                                  partial = TRUE)$Parameter)],
      H2a_B    = coef(res$m_h2a)["conditiongood"],
      H2a_p    = summary(res$m_h2a)$coefficients["conditiongood", "Pr(>|t|)"],
      H2b_B    = coef(res$m_h2b)["conditiongood"],
      H2b_p    = summary(res$m_h2b)$coefficients["conditiongood", "Pr(>|t|)"],
      H4_d     = effectsize::cohens_d(dq_mean ~ condition, data = df)$Cohens_d,
      H4_p     = res$t_h4$p.value,
      H5a_ACME = summary(res$med_diet)$d0,
      H5a_p    = summary(res$med_diet)$d0.p,
      H5b_ACME = summary(res$med_cook)$d0,
      H5b_p    = summary(res$med_cook)$d0.p,
      H6a_B    = coef(res$m_h6_diet)["llm_quality_index"],
      H6a_p    = summary(res$m_h6_diet)$coefficients["llm_quality_index", "Pr(>|t|)"],
      H6b_B    = coef(res$m_h6_stars)["llm_quality_index"],
      H6b_p    = summary(res$m_h6_stars)$coefficients["llm_quality_index", "Pr(>|t|)"],
      H6c_B    = coef(res$m_h6_cook)["llm_quality_index"],
      H6c_p    = summary(res$m_h6_cook)$coefficients["llm_quality_index", "Pr(>|t|)"]
    )
  }
  
  comparison <- bind_rows(
    extract_row(res_primary, df_primary, "PRIMARY"),
    extract_row(res_sens,    df_sens,    "SENS")
  ) %>%
    mutate(across(where(is.numeric), ~ round(.x, 4)))
  
  cat("\n\n", strrep("=", 70), "\n")
  cat("  PRIMARY vs. SENS - Direct comparison\n")
  cat(strrep("=", 70), "\n\n")
  print(as.data.frame(t(comparison)))
  
  write_csv(comparison, "output_comparison_PRIMARY_vs_SENS_StudyB.csv")
  cat("\nComparison table saved: output_comparison_PRIMARY_vs_SENS_StudyB.csv\n")
  
  invisible(comparison)
}

# Run comparison
comparison_table <- compare_primary_sens(
  results_primary, results_sens,
  df_primary, df_sens
)


# -----
# Visualisations (PRIMARY dataset only)

theme_studyb <- theme_minimal(base_size = 13) +
  theme(plot.title      = element_text(face = "bold", size = 14),
        plot.subtitle   = element_text(color = "grey40"),
        legend.position = "bottom",
        strip.text      = element_text(face = "bold"))

cond_colors <- c("good" = "#2196F3", "bad" = "#F44336")
cond_labels <- c("good" = "Good Communication", "bad" = "Bad Communication")

prepost_plot <- function(data, pre_var, post_var, title, y_limits = c(1, 7)) {
  data %>%
    group_by(condition) %>%
    summarise(pre     = mean(.data[[pre_var]],  na.rm = TRUE),
              post    = mean(.data[[post_var]], na.rm = TRUE),
              se_pre  = sd(.data[[pre_var]],   na.rm = TRUE) / sqrt(n()),
              se_post = sd(.data[[post_var]],  na.rm = TRUE) / sqrt(n()),
              .groups = "drop") %>%
    pivot_longer(c(pre, post), names_to = "time", values_to = "mean") %>%
    mutate(se   = ifelse(time == "pre", se_pre, se_post),
           time = factor(time, levels = c("pre", "post"), labels = c("Pre", "Post"))) %>%
    ggplot(aes(x = time, y = mean, group = condition, color = condition)) +
    geom_line(linewidth = 1.2) + geom_point(size = 3) +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se), width = 0.1) +
    scale_color_manual(values = cond_colors, labels = cond_labels) +
    scale_y_continuous(limits = y_limits, breaks = seq(y_limits[1], y_limits[2])) +
    labs(title = title, x = "Time", y = "Mean Rating", color = NULL) +
    theme_studyb
}

p_diet  <- prepost_plot(df_primary, "pre_diet_suitability", "post_diet_suitability",
                        "H1: Diet Suitability (1-7)", c(1, 7))
p_stars <- prepost_plot(df_primary, "pre_recipe_stars",     "post_recipe_stars",
                        "H2a: Star Rating (1-5)",   c(1, 5))
p_cook  <- prepost_plot(df_primary, "pre_cook_intent",      "post_cook_intent",
                        "H2b: Cooking Intention (1-7)", c(1, 7))

combined_prepost <- (p_diet | p_stars | p_cook) +
  plot_annotation(title = "Study B — Pre/Post Ratings by Communication Condition",
                  theme = theme(plot.title = element_text(face = "bold", size = 15)))

p_dq <- df_primary %>%
  select(condition, dq_clarity, dq_relevance, dq_respect, dq_logic, dq_coherence, dq_mean) %>%
  pivot_longer(-condition, names_to = "subscale", values_to = "rating") %>%
  group_by(condition, subscale) %>%
  summarise(M = mean(rating, na.rm = TRUE),
            SE = sd(rating, na.rm = TRUE) / sqrt(n()), .groups = "drop") %>%
  mutate(subscale = case_match(subscale,
                               "dq_clarity"   ~ "Clarity",   "dq_relevance" ~ "Relevance",
                               "dq_respect"   ~ "Respect",   "dq_logic"     ~ "Logic",
                               "dq_coherence" ~ "Coherence", "dq_mean"      ~ "Overall (mean)")) %>%
  ggplot(aes(x = subscale, y = M, fill = condition)) +
  geom_col(position = position_dodge(0.8), width = 0.7) +
  geom_errorbar(aes(ymin = M - SE, ymax = M + SE),
                position = position_dodge(0.8), width = 0.2) +
  scale_fill_manual(values = cond_colors, labels = cond_labels) +
  scale_y_continuous(limits = c(0, 7), breaks = 0:7) +
  labs(title = "H4: Perceived Dialog Quality by Sub-Scale",
       x = NULL, y = "Mean Rating (1-7)", fill = NULL) +
  theme_studyb + theme(axis.text.x = element_text(angle = 20, hjust = 1))

p_delta <- df_primary %>%
  select(condition, delta_diet, delta_stars, delta_cook, delta_save) %>%
  pivot_longer(-condition, names_to = "outcome", values_to = "delta") %>%
  mutate(outcome = case_match(outcome,
                              "delta_diet"  ~ "Diet Suitability", "delta_stars" ~ "Star Rating",
                              "delta_cook"  ~ "Cook Intention",   "delta_save"  ~ "Save Intention")) %>%
  ggplot(aes(x = condition, y = delta, fill = condition)) +
  geom_violin(alpha = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.2, outlier.shape = 21, outlier.size = 2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  facet_wrap(~outcome, scales = "free_y") +
  scale_fill_manual(values = cond_colors, labels = cond_labels) +
  scale_x_discrete(labels = cond_labels) +
  labs(title = "Delta Scores (Post - Pre) by Condition",
       x = NULL, y = "Delta Score", fill = NULL) +
  theme_studyb + theme(axis.text.x = element_text(angle = 20, hjust = 1))

p_h6 <- df_primary %>%
  ggplot(aes(x = llm_quality_index, y = delta_diet, color = condition)) +
  geom_jitter(width = 0.05, alpha = 0.6, size = 2) +
  geom_smooth(method = "lm", se = TRUE, aes(group = 1),
              color = "black", linewidth = 0.8) +
  scale_color_manual(values = cond_colors, labels = cond_labels) +
  labs(title = "H6: LLM Quality Index vs. Delta Diet Suitability",
       x = "LLM Quality Index (Study A)",
       y = "Delta Diet Suitability (post-pre)", color = NULL) +
  theme_studyb

dir.create("output_PRIMARY", showWarnings = FALSE)
ggsave("output_PRIMARY/studyB_plot_prepost.png",      combined_prepost, width=14, height=5,  dpi=180)
ggsave("output_PRIMARY/studyB_plot_dq_subscales.png", p_dq,             width=9,  height=5,  dpi=180)
ggsave("output_PRIMARY/studyB_plot_deltas.png",       p_delta,          width=12, height=7,  dpi=180)
ggsave("output_PRIMARY/studyB_plot_h6_scatter.png",   p_h6,             width=8,  height=5,  dpi=180)

cat("\nAll plots saved in output_PRIMARY/\n")
cat("\n=== Scrips completed ===\n")
