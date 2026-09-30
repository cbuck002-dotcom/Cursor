# ============================================================
# CRITICAL FIXES 1, 2, 3: Data Preparation and Variable Definition
# ============================================================
# This file contains corrected code sections to replace broken portions.
# Insert these sections in the order listed below in your main script.

# ============================================================
# FIX 1: Define 'wide' from raw_bls (Line ~550)
# ============================================================
# LOCATION: After load_bls_primary_or_api() returns, before line 550
# REPLACES: The broken wide.get() calls

def create_wide_pivot(raw_bls_df):
    """
    Convert raw BLS long format (series_id, date, value) to wide format.
    Index: date
    Columns: series_id
    Values: observation value
    """
    raw_bls_df = raw_bls_df.copy()
    raw_bls_df["date"] = pd.to_datetime(raw_bls_df["date"], errors="coerce")
    raw_bls_df = raw_bls_df.dropna(subset=["date"])
    
    wide = raw_bls_df.pivot_table(
        index="date",
        columns="series_id",
        values="value",
        aggfunc="first"  # In case of duplicate date/series_id pairs
    )
    return wide.sort_index()


# USAGE (insert after line ~850):
# ================================
# raw_bls, BLS_SOURCE_USED = load_bls_primary_or_api()
# wide = create_wide_pivot(raw_bls)
# 
# # Now the following will work:
# monthly["NY_CPI"] = wide.get(bcol("NY_CPI"))
# monthly["Philadelphia_CPI"] = wide.get(bcol("Philadelphia_CPI"))
# ... etc


# ============================================================
# FIX 2: Define feature_means and feature_stds (Line ~1350)
# ============================================================
# LOCATION: Before any recursive_forecast() call, after train_df is created
# REPLACES: Missing feature normalization variables

def compute_feature_statistics(train_dataframe, feature_list):
    """
    Compute training-sample means and standard deviations for feature normalization.
    These are used by recursive_forecast() and run_scenario() to standardize inputs.
    
    Parameters
    ----------
    train_dataframe : pd.DataFrame
        Training sample (typically train_df)
    feature_list : list
        List of feature column names (typically MODEL_FEATURES)
    
    Returns
    -------
    means : pd.Series
        Index: feature names, values: training means
    stds : pd.Series
        Index: feature names, values: training stds (with zero replaced by 1.0)
    """
    means = train_dataframe[feature_list].mean()
    stds = train_dataframe[feature_list].std(ddof=0).replace(0, 1.0)
    
    # Guard against all-NaN columns
    for col in feature_list:
        if pd.isna(means[col]):
            means[col] = 0.0
        if pd.isna(stds[col]) or stds[col] == 0:
            stds[col] = 1.0
    
    return means, stds


# USAGE (insert after train_df is created, around line ~900):
# ============================================================
# feature_means, feature_stds = compute_feature_statistics(train_df, MODEL_FEATURES)
#
# # Now recursive_forecast() and run_scenario() can use these:
# # (vals - feature_means[MODEL_FEATURES].to_numpy()) / feature_stds[MODEL_FEATURES].to_numpy()


# ============================================================
# FIX 3: Complete Data Preparation Pipeline (Line ~900-1000)
# ============================================================
# LOCATION: Before fitting any models
# REPLACES: Missing train/test split and data alignment

def prepare_training_validation_data(model_df, target_col="Inflation", 
                                      feature_list=None, train_end_date="2021-12-01"):
    """
    Split model data into training and validation (test) sets.
    Standardize features using training-sample statistics.
    Align all required columns and handle missing values.
    
    Parameters
    ----------
    model_df : pd.DataFrame
        Complete modeling dataframe with DatetimeIndex
    target_col : str
        Name of target variable column (default: "Inflation")
    feature_list : list
        List of feature column names. If None, uses MODEL_FEATURES.
    train_end_date : str or Timestamp
        Last date to include in training sample (YYYY-MM-DD)
    
    Returns
    -------
    train_df : pd.DataFrame
        Training sample (index <= train_end_date)
    test_df : pd.DataFrame
        Validation sample (index > train_end_date)
    X_train : np.ndarray
        Training features (n_train, n_features) - NOT INCLUDING intercept
    y_train : np.ndarray
        Training target (n_train,)
    X_test : np.ndarray
        Validation features (n_test, n_features) - NOT INCLUDING intercept
    feature_means : pd.Series
        Training-sample means for each feature
    feature_stds : pd.Series
        Training-sample stds for each feature
    """
    if feature_list is None:
        feature_list = MODEL_FEATURES
    
    train_end = pd.Timestamp(train_end_date)
    
    # Split into training and test
    train_df = model_df.loc[:train_end].copy()
    test_df = model_df.loc[train_end + pd.offsets.MonthBegin(1):].copy()
    
    # Require minimum training observations
    if len(train_df) < MIN_TRAIN_OBS:
        raise ValueError(
            f"Training set has {len(train_df)} observations, but MIN_TRAIN_OBS={MIN_TRAIN_OBS} required. "
            f"Adjust TRAIN_END or HISTORICAL data."
        )
    
    # Extract target and check for missing values
    if target_col not in train_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in model_df.")
    
    y_train = train_df[target_col].to_numpy(dtype=float)
    
    # Check for NaN in target
    if np.any(np.isnan(y_train)):
        raise ValueError(f"Training target '{target_col}' contains NaN values.")
    
    # Compute feature statistics from training sample ONLY
    means = train_df[feature_list].mean()
    stds = train_df[feature_list].std(ddof=0).replace(0, 1.0)
    
    # Guard against all-NaN columns
    for col in feature_list:
        if pd.isna(means[col]):
            raise ValueError(f"Feature '{col}' is all-NaN in training sample.")
        if pd.isna(stds[col]):
            stds[col] = 1.0
    
    # Standardize training features
    X_train = train_df[feature_list].to_numpy(dtype=float)
    X_train_std = (X_train - means.to_numpy()) / stds.to_numpy()
    
    # Standardize test features using TRAINING statistics (critical!)
    X_test = test_df[feature_list].to_numpy(dtype=float)
    X_test_std = (X_test - means.to_numpy()) / stds.to_numpy()
    
    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train_std,
        "y_train": y_train,
        "X_test": X_test_std,
        "feature_means": means,
        "feature_stds": stds,
    }


# ============================================================
# CONSOLIDATED INITIALIZATION BLOCK
# ============================================================
# INSERT THIS BLOCK around line ~900, after model_source is created
# and before dynamic_model = DynamicInflationStateSpace(...)


def initialize_model_pipeline():
    """
    Complete pipeline to prepare data and initialize model inputs.
    Run this ONCE before model estimation.
    """
    global raw_bls, BLS_SOURCE_USED, wide, train_df, test_df, X_train, y_train, X_test
    global feature_means, feature_stds, model_df
    
    # Step 1: Load BLS data and establish wide format
    print("\n[Step 1/5] Loading BLS data...")
    raw_bls, BLS_SOURCE_USED = load_bls_primary_or_api()
    print(f"  BLS source: {BLS_SOURCE_USED}")
    print(f"  Raw BLS rows: {len(raw_bls)}")
    
    wide = create_wide_pivot(raw_bls)
    print(f"  Wide format: {wide.shape} (dates × series)")
    
    # Step 2: Build monthly table (replaces the broken wide.get() calls)
    print("\n[Step 2/5] Building monthly aggregates from BLS...")
    model_df = monthly.copy() if 'monthly' in dir() else pd.DataFrame(index=wide.index)
    model_df["Date"] = model_df.index
    
    # Populate from wide using the bcol() lookup
    for name, sid in SERIES.items():
        if sid in wide.columns:
            model_df[name] = wide[sid]
    
    # Step 3: Ensure required columns exist
    print("\n[Step 3/5] Validating required columns...")
    required_raw = [
        "Custom_MA_CPI_Proxy", "Unemployment_MA", "WageGrowth_MA",
        "PPI_Growth", "WTI_Growth", "FedFunds", "ClaimsGrowth"
    ]
    missing = [c for c in required_raw if c not in model_df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")
    print(f"  All {len(required_raw)} required raw columns present.")
    
    # Step 4: Prepare training/validation split
    print("\n[Step 4/5] Splitting training/validation...")
    prep = prepare_training_validation_data(
        model_df,
        target_col="Inflation",
        feature_list=MODEL_FEATURES,
        train_end_date=TRAIN_END
    )
    
    train_df = prep["train_df"]
    test_df = prep["test_df"]
    X_train = prep["X_train"]
    y_train = prep["y_train"]
    X_test = prep["X_test"]
    feature_means = prep["feature_means"]
    feature_stds = prep["feature_stds"]
    
    print(f"  Training: {len(train_df)} obs ({train_df.index.min():%Y-%m} to {train_df.index.max():%Y-%m})")
    print(f"  Validation: {len(test_df)} obs ({test_df.index.min():%Y-%m} to {test_df.index.max():%Y-%m})")
    print(f"  Features: {X_train.shape[1]}")
    
    # Step 5: Create design matrix for state-space model (intercept + features)
    print("\n[Step 5/5] Creating state-space design matrix...")
    X_train_design = np.column_stack([np.ones(len(X_train)), X_train])
    
    print(f"  Design matrix shape: {X_train_design.shape}")
    print(f"  Expected (k_states={len(MODEL_FEATURES)+1}, nobs={len(X_train)}): (nobs, {len(MODEL_FEATURES)+1})")
    
    print("\n[Initialization complete]\n")
    
    return {
        "raw_bls": raw_bls,
        "wide": wide,
        "model_df": model_df,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "X_train_design": X_train_design,
        "feature_means": feature_means,
        "feature_stds": feature_stds,
        "BLS_SOURCE_USED": BLS_SOURCE_USED,
    }


# ============================================================
# HOW TO USE THESE FIXES IN YOUR SCRIPT
# ============================================================
# Replace the broken section (lines ~900-1000) with:
#
# # Initialize all data and prepare for model estimation
# init_result = initialize_model_pipeline()
#
# # Unpack returned dict into module globals
# raw_bls = init_result["raw_bls"]
# wide = init_result["wide"]
# model_df = init_result["model_df"]
# train_df = init_result["train_df"]
# test_df = init_result["test_df"]
# X_train = init_result["X_train"]
# y_train = init_result["y_train"]
# X_test = init_result["X_test"]
# X_train_design = init_result["X_train_design"]
# feature_means = init_result["feature_means"]
# feature_stds = init_result["feature_stds"]
# BLS_SOURCE_USED = init_result["BLS_SOURCE_USED"]
#
# # NOW YOU CAN SAFELY CREATE THE STATE-SPACE MODEL:
# dynamic_model = DynamicInflationStateSpace(y_train, X_train_design, feature_names=MODEL_FEATURES)
# fit_result = dynamic_model.fit(method="lbfgs", maxiter=1000, disp=False, cov_type="approx")


print("\n" + "="*80)
print("CRITICAL FIXES LOADED")
print("="*80)
print("Functions available:")
print("  - create_wide_pivot(raw_bls_df)")
print("  - compute_feature_statistics(train_df, feature_list)")
print("  - prepare_training_validation_data(model_df, target_col, feature_list, train_end_date)")
print("  - initialize_model_pipeline()")
print("\nCall initialize_model_pipeline() once before model estimation.")
print("="*80 + "\n")
