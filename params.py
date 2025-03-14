#Ridge Param Grid
ridge_param_grid={
    'alpha':[0.1,1.0,10.0],
    'fit_intercept':[True,False]
}

elasticnet_param_grid={
    'alpha':[0.1,1.0,10.0],
    'l1_ratio':[0.1,0.5,0.9],
    'fit_intercept':[True,False]
}


xgb_param_grid={
    'max_depth':[3,5,7],
    'n_estimators':[100,500,1000],
    'learning_rate':[0.01,0.1,0.3],
    'subsample':[0.5,0.7,0.9],
    'colsample_bytree':[0.5,0.7,0.9],
    'gamma':[0,1,5],
    'reg_alpha':[0,1,10],
    'reg_lambda':[0,1,10]
}