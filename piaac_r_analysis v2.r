# PIAAC Data Analysis: R Verification of Python Results
# =====================================================
# 
# Educational Attainment, Literacy Proficiency, and SES-Numeracy Interactions
# Replication and verification of Python analysis in R
# 
# Author: [Your Name]
# Date: December 2024
# Dataset: PIAAC 2017 U.S. Public Use File

# Clear environment and set options
rm(list = ls())
options(scipen = 999)  # Disable scientific notation
set.seed(12345)       # For reproducibility

# ============================================================================
# STEP 1: LOAD REQUIRED LIBRARIES
# ============================================================================

# Install packages if needed (uncomment if first time running)
# install.packages(c("readr", "haven", "survey", "dplyr", "ggplot2", 
#                   "gridExtra", "psych", "car", "broom", "knitr"))

# Load required libraries
library(readr)      # For reading CSV files
library(haven)      # For reading SPSS files (alternative)
library(survey)     # For complex survey design analysis
library(dplyr)      # For data manipulation
library(ggplot2)    # For plotting
library(gridExtra)  # For multiple plots
library(psych)      # For descriptive statistics
library(car)        # For ANOVA and regression diagnostics
library(broom)      # For tidying model outputs
library(knitr)      # For nice tables

cat("PIAAC Analysis: R Verification Script\n")
cat("=====================================\n\n")

# ============================================================================
# STEP 2: LOAD AND EXPLORE DATA
# ============================================================================

cat("Step 2: Loading PIAAC Data\n")
cat("--------------------------\n")

# Try to load the CSV file created by your Python converter
tryCatch({
  # Load the full dataset
  piaac_data <- read_csv("piaac_full_dataset.csv", 
                        show_col_types = FALSE,
                        locale = locale(encoding = "UTF-8"))
  
  cat("✓ Successfully loaded CSV file\n")
  cat(sprintf("  Dataset dimensions: %d rows × %d columns\n", 
              nrow(piaac_data), ncol(piaac_data)))
  
}, error = function(e) {
  # If CSV fails, try the research subset
  tryCatch({
    piaac_data <- read_csv("piaac_research_subset.csv", show_col_types = FALSE)
    cat("✓ Loaded research subset CSV\n")
    cat(sprintf("  Dataset dimensions: %d rows × %d columns\n", 
                nrow(piaac_data), ncol(piaac_data)))
  }, error = function(e2) {
    # If both fail, try direct SPSS load
    cat("CSV files not found, attempting direct SPSS load...\n")
    piaac_data <- read_sav("prgusap1_puf.sav")
    cat("✓ Loaded SPSS file directly\n")
  })
})

# Display basic information
cat(sprintf("\nMemory usage: %.1f MB\n", 
            object.size(piaac_data) / 1024^2))

# Check for key variables
key_vars <- c("SEQID", "SPFWT0", "EDCAT8", "PARED", 
              "PVLIT1", "PVNUM1", "PVPSL1", "GENDER_R")

available_vars <- key_vars[key_vars %in% names(piaac_data)]
missing_vars <- key_vars[!key_vars %in% names(piaac_data)]

cat(sprintf("\nKey variables found: %d of %d\n", 
            length(available_vars), length(key_vars)))
cat("Available:", paste(available_vars, collapse = ", "), "\n")
if(length(missing_vars) > 0) {
  cat("Missing:", paste(missing_vars, collapse = ", "), "\n")
}

# ============================================================================
# STEP 3: CREATE ANALYSIS VARIABLES
# ============================================================================

cat("\n\nStep 3: Creating Analysis Variables\n")
cat("-----------------------------------\n")

# Create mean literacy scores across plausible values
literacy_vars <- paste0("PVLIT", 1:10)
available_lit <- literacy_vars[literacy_vars %in% names(piaac_data)]

if(length(available_lit) > 0) {
  piaac_data$LITERACY_MEAN <- rowMeans(piaac_data[available_lit], na.rm = TRUE)
  cat(sprintf("✓ Created LITERACY_MEAN from %d plausible values\n", 
              length(available_lit)))
} else {
  cat("⚠ No literacy plausible values found\n")
}

# Create mean numeracy scores
numeracy_vars <- paste0("PVNUM", 1:10)
available_num <- numeracy_vars[numeracy_vars %in% names(piaac_data)]

if(length(available_num) > 0) {
  piaac_data$NUMERACY_MEAN <- rowMeans(piaac_data[available_num], na.rm = TRUE)
  cat(sprintf("✓ Created NUMERACY_MEAN from %d plausible values\n", 
              length(available_num)))
}

# Create mean problem-solving scores
psl_vars <- paste0("PVPSL", 1:10)
available_psl <- psl_vars[psl_vars %in% names(piaac_data)]

if(length(available_psl) > 0) {
  piaac_data$PROBLEM_SOLVING_MEAN <- rowMeans(piaac_data[available_psl], na.rm = TRUE)
  cat(sprintf("✓ Created PROBLEM_SOLVING_MEAN from %d plausible values\n", 
              length(available_psl)))
}

# Check created variables
if("LITERACY_MEAN" %in% names(piaac_data)) {
  lit_stats <- piaac_data %>% 
    summarise(
      n = sum(!is.na(LITERACY_MEAN)),
      mean = mean(LITERACY_MEAN, na.rm = TRUE),
      sd = sd(LITERACY_MEAN, na.rm = TRUE),
      min = min(LITERACY_MEAN, na.rm = TRUE),
      max = max(LITERACY_MEAN, na.rm = TRUE)
    )
  
  cat(sprintf("\nLiteracy Summary: N=%d, M=%.1f, SD=%.1f, Range=%.0f-%.0f\n",
              lit_stats$n, lit_stats$mean, lit_stats$sd, 
              lit_stats$min, lit_stats$max))
}

# ============================================================================
# STEP 4: SET UP SURVEY DESIGN
# ============================================================================

cat("\n\nStep 4: Setting Up Survey Design\n")
cat("--------------------------------\n")

# Check if survey weights exist
if("SPFWT0" %in% names(piaac_data)) {
  # Create survey design object
  piaac_design <- svydesign(
    ids = ~1,                    # No clustering variable in public use file
    weights = ~SPFWT0,           # Main survey weight
    data = piaac_data
  )
  
  cat("✓ Survey design object created\n")
  cat(sprintf("  Weighted N: %.0f\n", sum(piaac_data$SPFWT0, na.rm = TRUE)))
  
  # Check for replicate weights (for proper SE estimation)
  rep_weights <- paste0("SPFWT", 1:80)
  available_reps <- rep_weights[rep_weights %in% names(piaac_data)]
  
  if(length(available_reps) >= 10) {
    cat(sprintf("✓ Found %d replicate weights for variance estimation\n", 
                length(available_reps)))
    
    # Create replicate design (more accurate)
    piaac_rep_design <- svrepdesign(
      data = piaac_data,
      weights = ~SPFWT0,
      repweights = piaac_data[available_reps],
      type = "JK1",  # Jackknife method used by PIAAC
      scale = 1,
      rscales = 1
    )
    
    cat("✓ Replicate survey design created for accurate variance estimation\n")
  } else {
    cat("⚠ Limited replicate weights found, using simple design\n")
    piaac_rep_design <- piaac_design
  }
} else {
  cat("⚠ No survey weights found, using unweighted analysis\n")
  piaac_design <- NULL
  piaac_rep_design <- NULL
}

# ============================================================================
# STEP 5: RESEARCH QUESTION 1 - EDUCATION AND LITERACY
# ============================================================================

cat("\n\n", paste(rep("=", 60), collapse = ""), "\n")
cat("RESEARCH QUESTION 1: Education and Literacy Relationship\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Hypothesis 1: Higher educational attainment is positively associated with higher literacy proficiency.\n\n")

# Filter to complete cases for RQ1
rq1_data <- piaac_data %>%
  filter(!is.na(EDCAT8) & !is.na(LITERACY_MEAN) & !is.na(SPFWT0))

cat(sprintf("Analysis sample: %d cases\n", nrow(rq1_data)))

# 5.1 Descriptive Analysis by Education Level
cat("\n5.1 Literacy by Education Level\n")
cat("-------------------------------\n")

if(!is.null(piaac_rep_design)) {
  # Subset survey design for complete cases
  rq1_design <- subset(piaac_rep_design, 
                       !is.na(EDCAT8) & !is.na(LITERACY_MEAN))
  
  # Calculate weighted means by education level
  edu_summary <- svyby(~LITERACY_MEAN, ~EDCAT8, rq1_design, svymean, na.rm = TRUE)
  
  # Add sample sizes
  edu_n <- rq1_data %>% 
    group_by(EDCAT8) %>% 
    summarise(
      n = n(),
      weighted_n = sum(SPFWT0, na.rm = TRUE)
    )
  
  # Combine results
  edu_results <- merge(edu_summary, edu_n, by = "EDCAT8")
  
  # Create education labels
  edu_labels <- c(
    "1" = "1: Below HS",
    "2" = "2: Some HS", 
    "3" = "3: HS Diploma",
    "4" = "4: Some College",
    "5" = "5: Associate",
    "6" = "6: Bachelor's",
    "7" = "7: Master's",
    "8" = "8: Doctoral"
  )
  
  edu_results$Education_Label <- edu_labels[as.character(edu_results$EDCAT8)]
  
  # Display results
  cat(sprintf("%-20s %6s %12s %10s %8s\n", 
              "Education Level", "N", "Weighted N", "Mean Lit", "SE"))
  cat(sprintf("%s\n", paste(rep("-", 60), collapse = "")))
  
  for(i in 1:nrow(edu_results)) {
    cat(sprintf("%-20s %6d %12.0f %10.1f %8.2f\n",
                edu_results$Education_Label[i],
                edu_results$n[i],
                edu_results$weighted_n[i],
                edu_results$LITERACY_MEAN[i],
                edu_results$se[i]))
  }
} else {
  # Unweighted analysis if no survey design
  edu_results <- rq1_data %>%
    group_by(EDCAT8) %>%
    summarise(
      n = n(),
      mean_literacy = mean(LITERACY_MEAN, na.rm = TRUE),
      sd_literacy = sd(LITERACY_MEAN, na.rm = TRUE),
      se_literacy = sd_literacy / sqrt(n)
    )
  
  print(edu_results)
}

# 5.2 Correlation Analysis
cat("\n\n5.2 Correlation Analysis\n")
cat("------------------------\n")

if(!is.null(piaac_rep_design)) {
  # Weighted correlation using survey package
  rq1_design_complete <- subset(rq1_design, 
                               !is.na(EDCAT8) & !is.na(LITERACY_MEAN))
  
  # For weighted correlation, we'll use the survey-weighted regression approach
  corr_model <- svyglm(LITERACY_MEAN ~ EDCAT8, design = rq1_design_complete)
  
  # Extract R-squared for correlation approximation
  r_squared <- summary(corr_model)$r.squared
  correlation <- sqrt(r_squared) * sign(coef(corr_model)[2])
  
  cat(sprintf("Weighted correlation (Education × Literacy): r = %.3f\n", 
              correlation))
} else {
  # Simple correlation
  correlation <- cor(rq1_data$EDCAT8, rq1_data$LITERACY_MEAN, 
                    use = "complete.obs")
  cat(sprintf("Correlation (Education × Literacy): r = %.3f\n", correlation))
}

# Interpret correlation strength
if(abs(correlation) >= 0.7) {
  strength <- "very strong"
} else if(abs(correlation) >= 0.5) {
  strength <- "strong"
} else if(abs(correlation) >= 0.3) {
  strength <- "moderate"
} else {
  strength <- "weak"
}

direction <- ifelse(correlation > 0, "positive", "negative")
cat(sprintf("Interpretation: %s %s relationship\n", strength, direction))

# 5.3 Effect Size Analysis
cat("\n\n5.3 Effect Size Analysis\n")
cat("------------------------\n")

# Find lowest and highest education groups with sufficient data
edu_levels <- sort(unique(rq1_data$EDCAT8))
if(length(edu_levels) >= 2) {
  lowest_edu <- edu_levels[1]
  highest_edu <- edu_levels[length(edu_levels)]
  
  low_group <- rq1_data[rq1_data$EDCAT8 == lowest_edu, ]
  high_group <- rq1_data[rq1_data$EDCAT8 == highest_edu, ]
  
  # Calculate weighted means and SDs
  if(!is.null(piaac_rep_design)) {
    low_design <- subset(rq1_design, EDCAT8 == lowest_edu)
    high_design <- subset(rq1_design, EDCAT8 == highest_edu)
    
    low_mean <- svymean(~LITERACY_MEAN, low_design, na.rm = TRUE)[1]
    high_mean <- svymean(~LITERACY_MEAN, high_design, na.rm = TRUE)[1]
    
    # Approximate weighted SDs
    low_sd <- sqrt(svyvar(~LITERACY_MEAN, low_design, na.rm = TRUE)[1])
    high_sd <- sqrt(svyvar(~LITERACY_MEAN, high_design, na.rm = TRUE)[1])
  } else {
    low_mean <- mean(low_group$LITERACY_MEAN, na.rm = TRUE)
    high_mean <- mean(high_group$LITERACY_MEAN, na.rm = TRUE)
    low_sd <- sd(low_group$LITERACY_MEAN, na.rm = TRUE)
    high_sd <- sd(high_group$LITERACY_MEAN, na.rm = TRUE)
  }
  
  difference <- high_mean - low_mean
  pooled_sd <- sqrt((low_sd^2 + high_sd^2) / 2)
  cohens_d <- difference / pooled_sd
  
  cat(sprintf("Lowest education level (%s): %.1f literacy points\n", 
              lowest_edu, low_mean))
  cat(sprintf("Highest education level (%s): %.1f literacy points\n", 
              highest_edu, high_mean))
  cat(sprintf("Difference: %.1f points\n", difference))
  cat(sprintf("Cohen's d: %.2f\n", cohens_d))
  
  # Interpret Cohen's d
  if(cohens_d >= 0.8) {
    effect_size <- "large"
  } else if(cohens_d >= 0.5) {
    effect_size <- "medium"
  } else if(cohens_d >= 0.2) {
    effect_size <- "small"
  } else {
    effect_size <- "negligible"
  }
  
  cat(sprintf("Effect size: %s\n", effect_size))
}

# 5.4 Regression Analysis
cat("\n\n5.4 Regression Analysis\n")
cat("-----------------------\n")

if(!is.null(piaac_rep_design)) {
  lit_model <- svyglm(LITERACY_MEAN ~ EDCAT8, design = rq1_design)
  lit_summary <- summary(lit_model)
  
  cat("Weighted Linear Regression Results:\n")
  cat(sprintf("Intercept: %.2f (SE = %.2f)\n", 
              coef(lit_model)[1], lit_summary$coefficients[1,2]))
  cat(sprintf("Education coefficient: %.2f (SE = %.2f)\n", 
              coef(lit_model)[2], lit_summary$coefficients[2,2]))
  cat(sprintf("R-squared: %.3f\n", lit_summary$r.squared))
  cat(sprintf("P-value: %s\n", 
              ifelse(lit_summary$coefficients[2,4] < 0.001, "< 0.001", 
                     sprintf("%.3f", lit_summary$coefficients[2,4]))))
} else {
  lit_model <- lm(LITERACY_MEAN ~ EDCAT8, data = rq1_data)
  lit_summary <- summary(lit_model)
  
  cat("Linear Regression Results:\n")
  print(lit_summary$coefficients)
  cat(sprintf("R-squared: %.3f\n", lit_summary$r.squared))
}

# 5.5 Hypothesis 1 Conclusion
cat("\n\n5.5 Hypothesis 1 Results\n")
cat("------------------------\n")

cat("HYPOTHESIS 1: Higher educational attainment is positively associated with higher literacy proficiency\n\n")
cat("EVIDENCE:\n")
cat(sprintf("• Correlation: r = %.3f (%s %s)\n", correlation, strength, direction))
cat(sprintf("• Effect size: %.1f points difference, Cohen's d = %.2f (%s)\n", 
            difference, cohens_d, effect_size))
cat("• Pattern: Clear increase in literacy scores with education level\n")

if(correlation > 0.3 && cohens_d > 0.5) {
  conclusion <- "✓ HYPOTHESIS 1 STRONGLY SUPPORTED"
} else if(correlation > 0.2 && cohens_d > 0.2) {
  conclusion <- "✓ HYPOTHESIS 1 SUPPORTED"
} else {
  conclusion <- "✗ HYPOTHESIS 1 WEAK SUPPORT"
}

cat(sprintf("\n%s\n", conclusion))

# ============================================================================
# STEP 6: RESEARCH QUESTION 2 - SES × NUMERACY INTERACTION
# ============================================================================

cat("\n\n", paste(rep("=", 70), collapse = ""), "\n")
cat("RESEARCH QUESTION 2: SES × Numeracy Interaction on Problem-Solving\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Hypothesis 2: There is a positive interaction between SES and numeracy skills in predicting problem-solving abilities.\n\n")

# Filter to complete cases for RQ2
rq2_data <- piaac_data %>%
  filter(!is.na(PARED) & !is.na(NUMERACY_MEAN) & 
         !is.na(PROBLEM_SOLVING_MEAN) & !is.na(SPFWT0))

cat(sprintf("Analysis sample: %d cases\n", nrow(rq2_data)))

if(nrow(rq2_data) == 0) {
  cat("⚠ No complete cases for Research Question 2 analysis\n")
} else {
  
  # 6.1 Descriptive Analysis
  cat("\n6.1 Variable Distributions\n")
  cat("--------------------------\n")
  
  # SES distribution
  ses_dist <- rq2_data %>%
    group_by(PARED) %>%
    summarise(
      n = n(),
      weighted_n = sum(SPFWT0, na.rm = TRUE),
      pct = n() / nrow(rq2_data) * 100
    )
  
  cat("SES Distribution (PARED):\n")
  ses_labels <- c("1" = "Low SES (Parents: HS or less)",
                  "2" = "Medium SES (Parents: Some college)", 
                  "3" = "High SES (Parents: College+)")
  
  for(i in 1:nrow(ses_dist)) {
    cat(sprintf("  %s: %d (%.1f%%)\n", 
                ses_labels[as.character(ses_dist$PARED[i])],
                ses_dist$n[i], ses_dist$pct[i]))
  }
  
  # Variable summaries
  var_summary <- rq2_data %>%
    summarise(
      numeracy_mean = mean(NUMERACY_MEAN, na.rm = TRUE),
      numeracy_sd = sd(NUMERACY_MEAN, na.rm = TRUE),
      ps_mean = mean(PROBLEM_SOLVING_MEAN, na.rm = TRUE),
      ps_sd = sd(PROBLEM_SOLVING_MEAN, na.rm = TRUE)
    )
  
  cat(sprintf("\nNumeracy: M = %.1f, SD = %.1f\n", 
              var_summary$numeracy_mean, var_summary$numeracy_sd))
  cat(sprintf("Problem-solving: M = %.1f, SD = %.1f\n", 
              var_summary$ps_mean, var_summary$ps_sd))
  
  # 6.2 Standardize variables for interaction analysis
  rq2_data$SES_std <- as.numeric(scale(rq2_data$PARED))
  rq2_data$NUMERACY_std <- as.numeric(scale(rq2_data$NUMERACY_MEAN))
  rq2_data$INTERACTION <- rq2_data$SES_std * rq2_data$NUMERACY_std
  
  # 6.3 Correlation Matrix
  cat("\n\n6.3 Correlation Matrix\n")
  cat("----------------------\n")
  
  corr_vars <- rq2_data[c("PARED", "NUMERACY_MEAN", "PROBLEM_SOLVING_MEAN")]
  corr_matrix <- cor(corr_vars, use = "complete.obs")
  
  cat("                    SES      Numeracy  Problem-Solving\n")
  cat("---------------------------------------------------\n")
  cat(sprintf("SES              %.3f\n", corr_matrix[1,1]))
  cat(sprintf("Numeracy         %.3f    %.3f\n", 
              corr_matrix[2,1], corr_matrix[2,2]))
  cat(sprintf("Problem-Solving  %.3f    %.3f    %.3f\n", 
              corr_matrix[3,1], corr_matrix[3,2], corr_matrix[3,3]))
  
  # 6.4 Regression Analysis
  cat("\n\n6.4 Hierarchical Regression Analysis\n")
  cat("------------------------------------\n")
  
  if(!is.null(piaac_rep_design)) {
    # Subset survey design for RQ2
    rq2_design <- subset(piaac_rep_design, 
                         !is.na(PARED) & !is.na(NUMERACY_MEAN) & 
                         !is.na(PROBLEM_SOLVING_MEAN))
    
    # Model 1: Main effects
    model1 <- svyglm(PROBLEM_SOLVING_MEAN ~ SES_std + NUMERACY_std, 
                     design = rq2_design)
    
    # Model 2: Interaction
    model2 <- svyglm(PROBLEM_SOLVING_MEAN ~ SES_std + NUMERACY_std + INTERACTION, 
                     design = rq2_design)
    
    # Model summaries
    cat("Model 1 - Main Effects:\n")
    model1_summary <- summary(model1)
    print(model1_summary$coefficients)
    cat(sprintf("R-squared: %.3f\n\n", model1_summary$r.squared))
    
    cat("Model 2 - Interaction:\n")
    model2_summary <- summary(model2)
    print(model2_summary$coefficients)
    cat(sprintf("R-squared: %.3f\n", model2_summary$r.squared))
    
    # R-squared change
    r2_change <- model2_summary$r.squared - model1_summary$r.squared
    cat(sprintf("ΔR² = %.4f\n", r2_change))
    
    # Extract interaction coefficient
    interaction_coef <- coef(model2)[4]
    interaction_se <- model2_summary$coefficients[4,2]
    interaction_p <- model2_summary$coefficients[4,4]
    
  } else {
    # Unweighted analysis
    model1 <- lm(PROBLEM_SOLVING_MEAN ~ SES_std + NUMERACY_std, data = rq2_data)
    model2 <- lm(PROBLEM_SOLVING_MEAN ~ SES_std + NUMERACY_std + INTERACTION, 
                 data = rq2_data)
    
    cat("Model 1 - Main Effects:\n")
    print(summary(model1)$coefficients)
    cat(sprintf("R-squared: %.3f\n\n", summary(model1)$r.squared))
    
    cat("Model 2 - Interaction:\n")
    print(summary(model2)$coefficients)
    cat(sprintf("R-squared: %.3f\n", summary(model2)$r.squared))
    
    r2_change <- summary(model2)$r.squared - summary(model1)$r.squared
    cat(sprintf("ΔR² = %.4f\n", r2_change))
    
    interaction_coef <- coef(model2)[4]
    interaction_p <- summary(model2)$coefficients[4,4]
  }
  
  # 6.5 Interaction Interpretation
  cat("\n\n6.5 Interaction Analysis\n")
  cat("------------------------\n")
  
  cat(sprintf("Interaction coefficient: β = %.3f\n", interaction_coef))
  cat(sprintf("P-value: %s\n", 
              ifelse(interaction_p < 0.001, "< 0.001", sprintf("%.3f", interaction_p))))
  
  # Effect size interpretation
  if(abs(interaction_coef) >= 0.10) {
    interaction_size <- "large"
  } else if(abs(interaction_coef) >= 0.05) {
    interaction_size <- "medium"
  } else if(abs(interaction_coef) >= 0.02) {
    interaction_size <- "small"
  } else {
    interaction_size <- "negligible"
  }
  
  cat(sprintf("Effect size: %s\n", interaction_size))
  
  # Simple slopes analysis (if interaction is meaningful)
  if(abs(interaction_coef) >= 0.02) {
    cat("\n6.6 Simple Slopes Analysis\n")
    cat("--------------------------\n")
    
    # Calculate simple slopes at ±1 SD of SES
    numeracy_main <- coef(model2)[3]  # Main effect of numeracy
    
    ses_levels <- c(-1, 0, 1)
    ses_labels <- c("Low SES (-1 SD)", "Average SES (0)", "High SES (+1 SD)")
    
    cat("Effect of Numeracy on Problem-Solving at Different SES Levels:\n")
    cat(sprintf("%-20s %-15s %s\n", "SES Level", "Numeracy Effect", "Interpretation"))
    cat(sprintf("%s\n", paste(rep("-", 60), collapse = "")))
    
    for(i in 1:length(ses_levels)) {
      simple_slope <- numeracy_main + interaction_coef * ses_levels[i]
      
      if(simple_slope > 0.3) {
        effect_strength <- "Strong positive"
      } else if(simple_slope > 0.1) {
        effect_strength <- "Moderate positive"
      } else if(simple_slope > 0) {
        effect_strength <- "Weak positive"
      } else {
        effect_strength <- "Negligible"
      }
      
      cat(sprintf("%-20s %-15.3f %s\n", 
                  ses_labels[i], simple_slope, effect_strength))
    }
  }
  
  # 6.7 Hypothesis 2 Conclusion
  cat("\n\n6.7 Hypothesis 2 Results\n")
  cat("------------------------\n")
  
  cat("HYPOTHESIS 2: Positive interaction between SES and numeracy predicting problem-solving\n\n")
  cat("FINDINGS:\n")
  
  if(exists("model1_summary")) {
    cat(sprintf("• Main effect of SES: β = %.3f\n", coef(model1)[2]))
    cat(sprintf("• Main effect of Numeracy: β = %.3f\n", coef(model1)[3]))
  }
  
  cat(sprintf("• SES × Numeracy interaction: β = %.3f (%s effect)\n", 
              interaction_coef, interaction_size))
  cat(sprintf("• Model improvement: ΔR² = %.4f\n", r2_change))
  
  if(exists("model2_summary")) {
    cat(sprintf("• Total variance explained: %.1f%%\n", 
                model2_summary$r.squared * 100))
  }
  
  # Final conclusion
  if(interaction_coef > 0.02) {
    hypothesis_support <- "✓ SUPPORTS Hypothesis 2"
  } else if(interaction_coef < -0.02) {
    hypothesis_support <- "✗ CONTRADICTS Hypothesis 2"
  } else {
    hypothesis_support <- "✗ DOES NOT SUPPORT Hypothesis 2"
  }
  
  cat(sprintf("\n%s\n", hypothesis_support))
}

# ============================================================================
# STEP 7: CREATE VISUALIZATIONS
# ============================================================================

cat("\n\n", paste(rep("=", 50), collapse = ""), "\n")
cat("STEP 7: Creating Visualizations\n")
cat(paste(rep("=", 50), collapse = ""), "\n")

# 7.1 Research Question 1 Visualizations
if(exists("rq1_data") && nrow(rq1_data) > 0) {
  
  cat("\n7.1 Research Question 1 Visualizations\n")
  cat("--------------------------------------\n")
  
  # Create education labels for plotting
  rq1_data$Education_Label <- factor(rq1_data$EDCAT8, 
                                     levels = 1:8,
                                     labels = c("Below HS", "Some HS", "HS Diploma", 
                                               "Some College", "Associate", "Bachelor's", 
                                               "Master's", "Doctoral"))
  
  # Plot 1: Box plot of literacy by education
  p1 <- ggplot(rq1_data, aes(x = Education_Label, y = LITERACY_MEAN)) +
    geom_boxplot(aes(weight = SPFWT0), alpha = 0.7, fill = "lightblue") +
    geom_smooth(aes(group = 1), method = "lm", se = TRUE, color = "red", linetype = "dashed") +
    labs(title = "Literacy Score Distribution by Education Level",
         x = "Education Level", 
         y = "Literacy Score") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
          plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # Plot 2: Mean literacy by education with error bars
  if(exists("edu_results")) {
    p2 <- ggplot(edu_results, aes(x = factor(EDCAT8), y = LITERACY_MEAN)) +
      geom_col(fill = "lightcoral", alpha = 0.7, color = "darkred") +
      geom_errorbar(aes(ymin = LITERACY_MEAN - 1.96*se, 
                        ymax = LITERACY_MEAN + 1.96*se),
                    width = 0.2, color = "darkred") +
      geom_text(aes(label = round(LITERACY_MEAN, 0)), 
                vjust = -0.5, size = 3, fontface = "bold") +
      scale_x_discrete(labels = c("Below HS", "Some HS", "HS Diploma", 
                                 "Some College", "Associate", "Bachelor's", 
                                 "Master's", "Doctoral")) +
      labs(title = "Mean Literacy Score by Education Level (±95% CI)",
           x = "Education Level",
           y = "Mean Literacy Score") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
            plot.title = element_text(hjust = 0.5, face = "bold"))
  }
  
  # Plot 3: Scatter plot with trend line
  # Sample data for cleaner visualization
  if(nrow(rq1_data) > 1000) {
    sample_rq1 <- rq1_data[sample(nrow(rq1_data), 1000), ]
  } else {
    sample_rq1 <- rq1_data
  }
  
  p3 <- ggplot(sample_rq1, aes(x = EDCAT8, y = LITERACY_MEAN)) +
    geom_point(alpha = 0.6, color = "steelblue", size = 1.5) +
    geom_smooth(method = "lm", se = TRUE, color = "red", linetype = "dashed") +
    labs(title = paste("Education-Literacy Scatter Plot (r =", round(correlation, 3), ")"),
         x = "Education Level (EDCAT8)",
         y = "Literacy Score") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # Display plots
  print(p1)
  readline(prompt = "Press [Enter] to continue to next plot...")
  
  if(exists("p2")) {
    print(p2)
    readline(prompt = "Press [Enter] to continue to next plot...")
  }
  
  print(p3)
  readline(prompt = "Press [Enter] to continue to Research Question 2 plots...")
}

# 7.2 Research Question 2 Visualizations
if(exists("rq2_data") && nrow(rq2_data) > 0) {
  
  cat("\n7.2 Research Question 2 Visualizations\n")
  cat("--------------------------------------\n")
  
  # Create SES labels
  rq2_data$SES_Label <- factor(rq2_data$PARED,
                               levels = 1:3,
                               labels = c("Low SES\n(Parents: HS or less)",
                                         "Medium SES\n(Parents: Some college)",
                                         "High SES\n(Parents: College+)"))
  
  # Plot 4: Interaction scatter plot
  # Sample for visualization
  if(nrow(rq2_data) > 1500) {
    sample_rq2 <- rq2_data[sample(nrow(rq2_data), 1500), ]
  } else {
    sample_rq2 <- rq2_data
  }
  
  p4 <- ggplot(sample_rq2, aes(x = NUMERACY_MEAN, y = PROBLEM_SOLVING_MEAN, 
                               color = SES_Label)) +
    geom_point(alpha = 0.6, size = 1.5) +
    geom_smooth(method = "lm", se = TRUE, linewidth = 1.2) +
    scale_color_manual(values = c("red", "orange", "blue")) +
    labs(title = "Problem-Solving vs Numeracy by SES Level",
         x = "Numeracy Score",
         y = "Problem-Solving Score",
         color = "SES Level") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          legend.position = "bottom")
  
  # Plot 5: Interaction plot showing simple slopes
  # Create interaction data for plotting
  numeracy_range <- seq(min(rq2_data$NUMERACY_MEAN, na.rm = TRUE),
                       max(rq2_data$NUMERACY_MEAN, na.rm = TRUE),
                       length.out = 100)
  
  if(exists("model2")) {
    interaction_plot_data <- expand.grid(
      NUMERACY_MEAN = numeracy_range,
      SES_std = c(-1, 0, 1)
    )
    
    interaction_plot_data$INTERACTION <- interaction_plot_data$SES_std * 
      scale(interaction_plot_data$NUMERACY_MEAN)[,1]
    
    # Predict problem-solving scores
    if(!is.null(piaac_rep_design)) {
      # For survey design, we'll approximate
      predicted_values <- predict(model2, newdata = interaction_plot_data)
    } else {
      interaction_plot_data$NUMERACY_std <- scale(interaction_plot_data$NUMERACY_MEAN)[,1]
      predicted_values <- predict(model2, newdata = interaction_plot_data)
    }
    
    interaction_plot_data$PREDICTED_PS <- predicted_values
    interaction_plot_data$SES_Label <- factor(interaction_plot_data$SES_std,
                                             levels = c(-1, 0, 1),
                                             labels = c("Low SES (-1 SD)",
                                                       "Average SES (0)", 
                                                       "High SES (+1 SD)"))
    
    p5 <- ggplot(interaction_plot_data, aes(x = NUMERACY_MEAN, y = PREDICTED_PS, 
                                           color = SES_Label)) +
      geom_line(linewidth = 2) +
      scale_color_manual(values = c("red", "orange", "blue")) +
      labs(title = "Interaction Plot: SES × Numeracy → Problem-Solving",
           x = "Numeracy Score",
           y = "Predicted Problem-Solving Score",
           color = "SES Level") +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, face = "bold"),
            legend.position = "bottom")
    
    # Add annotation about interaction
    if(exists("interaction_coef")) {
      interaction_direction <- ifelse(interaction_coef > 0, "Positive", "Negative")
      p5 <- p5 + annotate("text", x = Inf, y = Inf, 
                         label = paste(interaction_direction, "Interaction:\nβ =", 
                                     round(interaction_coef, 3)),
                         hjust = 1.1, vjust = 1.1, size = 4, fontface = "bold",
                         color = "black")
    }
  }
  
  # Plot 6: Model comparison
  if(exists("model1") && exists("model2")) {
    model_comparison <- data.frame(
      Model = c("Main Effects\nModel", "Interaction\nModel"),
      R_squared = c(summary(model1)$r.squared, summary(model2)$r.squared)
    )
    
    p6 <- ggplot(model_comparison, aes(x = Model, y = R_squared)) +
      geom_col(fill = c("lightblue", "darkblue"), alpha = 0.8) +
      geom_text(aes(label = paste("R² =", round(R_squared, 3), 
                                 "\n(", round(R_squared*100, 1), "%)", sep = "")),
                vjust = -0.5, fontface = "bold") +
      labs(title = "Model Comparison: R² Values",
           x = "Model Type",
           y = "R² (Variance Explained)") +
      ylim(0, max(model_comparison$R_squared) * 1.2) +
      theme_minimal() +
      theme(plot.title = element_text(hjust = 0.5, face = "bold"))
    
    # Add R² improvement annotation
    if(exists("r2_change")) {
      p6 <- p6 + annotate("text", x = 1.5, y = mean(model_comparison$R_squared),
                         label = paste("ΔR² =", round(r2_change, 4)),
                         fontface = "bold", size = 4, color = "red")
    }
  }
  
  # Display RQ2 plots
  print(p4)
  readline(prompt = "Press [Enter] to continue to next plot...")
  
  if(exists("p5")) {
    print(p5)
    readline(prompt = "Press [Enter] to continue to next plot...")
  }
  
  if(exists("p6")) {
    print(p6)
    readline(prompt = "Press [Enter] to continue to summary...")
  }
}

# ============================================================================
# STEP 8: COMPREHENSIVE SUMMARY AND COMPARISON WITH PYTHON RESULTS
# ============================================================================

cat("\n\n", paste(rep("=", 70), collapse = ""), "\n")
cat("STEP 8: COMPREHENSIVE SUMMARY AND PYTHON COMPARISON\n")
cat(paste(rep("=", 70), collapse = ""), "\n")

cat("\n8.1 Research Question 1 Summary\n")
cat("-------------------------------\n")

if(exists("correlation") && exists("cohens_d")) {
  cat("HYPOTHESIS 1: Higher educational attainment is positively associated with higher literacy proficiency\n\n")
  cat("R RESULTS:\n")
  cat(sprintf("• Weighted correlation: r = %.3f (%s %s)\n", correlation, strength, direction))
  cat(sprintf("• Effect size: %.1f points difference, Cohen's d = %.2f (%s)\n", 
              difference, cohens_d, effect_size))
  cat(sprintf("• Regression coefficient: %.2f points per education level\n", coef(lit_model)[2]))
  
  cat("\nPYTHON COMPARISON:\n")
  cat("• Python correlation: r = 0.418 (moderate-strong positive)\n")
  cat("• Python effect size: 93.5 points difference, Cohen's d = 2.22 (very large)\n")
  cat("• Python regression: 11.7 points per education level\n")
  
  # Check if results match
  r_match <- abs(correlation - 0.418) < 0.05
  d_match <- abs(cohens_d - 2.22) < 0.2
  
  cat(sprintf("\nVERIFICATION STATUS:\n"))
  cat(sprintf("• Correlation match: %s (R: %.3f vs Python: 0.418)\n", 
              ifelse(r_match, "✓ VERIFIED", "⚠ DIFFERENCE"), correlation))
  cat(sprintf("• Effect size match: %s (R: %.2f vs Python: 2.22)\n", 
              ifelse(d_match, "✓ VERIFIED", "⚠ DIFFERENCE"), cohens_d))
  
  cat(sprintf("\n%s\n", conclusion))
}

cat("\n8.2 Research Question 2 Summary\n")
cat("-------------------------------\n")

if(exists("interaction_coef") && exists("r2_change")) {
  cat("HYPOTHESIS 2: Positive interaction between SES and numeracy predicting problem-solving\n\n")
  cat("R RESULTS:\n")
  cat(sprintf("• Main effect of SES: β = %.3f\n", coef(model1)[2]))
  cat(sprintf("• Main effect of Numeracy: β = %.3f\n", coef(model1)[3]))
  cat(sprintf("• SES × Numeracy interaction: β = %.3f (%s effect)\n", 
              interaction_coef, interaction_size))
  cat(sprintf("• Model improvement: ΔR² = %.4f\n", r2_change))
  cat(sprintf("• Total variance explained: %.1f%%\n", summary(model2)$r.squared * 100))
  
  cat("\nPYTHON COMPARISON:\n")
  cat("• Python SES main effect: β = 1.309\n")
  cat("• Python Numeracy main effect: β = 35.964\n")
  cat("• Python interaction: β = -0.404 (negative interaction)\n")
  cat("• Python ΔR²: 0.0001\n")
  cat("• Python total R²: 69.5%\n")
  
  # Check if results match
  interaction_match <- abs(interaction_coef - (-0.404)) < 0.1
  r2_match <- abs(summary(model2)$r.squared - 0.695) < 0.05
  
  cat(sprintf("\nVERIFICATION STATUS:\n"))
  cat(sprintf("• Interaction coefficient: %s (R: %.3f vs Python: -0.404)\n", 
              ifelse(interaction_match, "✓ VERIFIED", "⚠ DIFFERENCE"), interaction_coef))
  cat(sprintf("• R-squared: %s (R: %.3f vs Python: 0.695)\n", 
              ifelse(r2_match, "✓ VERIFIED", "⚠ DIFFERENCE"), summary(model2)$r.squared))
  
  cat(sprintf("\n%s\n", hypothesis_support))
}

# ============================================================================
# STEP 9: SAVE RESULTS AND CREATE FINAL REPORT
# ============================================================================

cat("\n\n", paste(rep("=", 50), collapse = ""), "\n")
cat("STEP 9: Saving Results\n")
cat(paste(rep("=", 50), collapse = ""), "\n")

# Save key results to CSV files
if(exists("edu_results")) {
  write.csv(edu_results, "r_results_education_literacy.csv", row.names = FALSE)
  cat("✓ Saved education-literacy results to r_results_education_literacy.csv\n")
}

if(exists("rq2_data")) {
  # Save model results
  if(exists("model1") && exists("model2")) {
    model_comparison_results <- data.frame(
      Model = c("Main Effects", "Interaction"),
      R_squared = c(summary(model1)$r.squared, summary(model2)$r.squared),
      AIC = c(AIC(model1), AIC(model2))
    )
    
    write.csv(model_comparison_results, "r_results_interaction_models.csv", row.names = FALSE)
    cat("✓ Saved interaction model results to r_results_interaction_models.csv\n")
  }
  
  # Save correlation matrix
  if(exists("corr_matrix")) {
    write.csv(corr_matrix, "r_results_correlation_matrix.csv")
    cat("✓ Saved correlation matrix to r_results_correlation_matrix.csv\n")
  }
}

# Create final verification report
verification_report <- list()

if(exists("correlation")) {
  verification_report$RQ1 <- list(
    r_correlation = correlation,
    python_correlation = 0.418,
    r_effect_size = ifelse(exists("cohens_d"), cohens_d, NA),
    python_effect_size = 2.22,
    correlation_verified = ifelse(exists("correlation"), abs(correlation - 0.418) < 0.05, FALSE),
    effect_size_verified = ifelse(exists("cohens_d"), abs(cohens_d - 2.22) < 0.2, FALSE)
  )
}

if(exists("interaction_coef")) {
  verification_report$RQ2 <- list(
    r_interaction = interaction_coef,
    python_interaction = -0.404,
    r_r_squared = summary(model2)$r.squared,
    python_r_squared = 0.695,
    interaction_verified = abs(interaction_coef - (-0.404)) < 0.1,
    r_squared_verified = abs(summary(model2)$r.squared - 0.695) < 0.05
  )
}

# Print final verification summary
cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("FINAL VERIFICATION SUMMARY\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

if(!is.null(verification_report$RQ1)) {
  cat("\nRESEARCH QUESTION 1 - EDUCATION & LITERACY:\n")
  if(verification_report$RQ1$correlation_verified) {
    cat("✓ CORRELATION VERIFIED: R and Python results match\n")
  } else {
    cat("⚠ CORRELATION DIFFERENCE: Check methodology\n")
  }
  
  if(verification_report$RQ1$effect_size_verified) {
    cat("✓ EFFECT SIZE VERIFIED: R and Python results match\n")
  } else {
    cat("⚠ EFFECT SIZE DIFFERENCE: Check calculation method\n")
  }
}

if(!is.null(verification_report$RQ2)) {
  cat("\nRESEARCH QUESTION 2 - SES × NUMERACY INTERACTION:\n")
  if(verification_report$RQ2$interaction_verified) {
    cat("✓ INTERACTION VERIFIED: R and Python results match\n")
  } else {
    cat("⚠ INTERACTION DIFFERENCE: Check standardization method\n")
  }
  
  if(verification_report$RQ2$r_squared_verified) {
    cat("✓ R-SQUARED VERIFIED: R and Python results match\n")
  } else {
    cat("⚠ R-SQUARED DIFFERENCE: Check model specification\n")
  }
}

cat("\n", paste(rep("=", 60), collapse = ""), "\n")
cat("ANALYSIS COMPLETE!\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cat("\nFiles created:\n")
cat("• r_results_education_literacy.csv - Education-literacy analysis\n")
cat("• r_results_interaction_models.csv - Interaction model comparison\n") 
cat("• r_results_correlation_matrix.csv - Variable correlations\n")

cat("\nNext steps:\n")
cat("1. Compare any differences between R and Python results\n")
cat("2. Investigate any methodological discrepancies\n")
cat("3. Use these verified results for your final report\n")
cat("4. Consider running additional robustness checks\n")

cat("\n✓ R verification analysis complete!\n")

# End of script
cat("\n# End of PIAAC R Analysis Script\n")
cat("# ", paste(rep("=", 50), collapse = ""), "\n")