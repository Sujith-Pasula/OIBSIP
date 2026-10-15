from pathlib import Path
import json, urllib.request, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'housing.csv'; OUT=ROOT/'outputs'; CHARTS=OUT/'charts'; REPORTS=OUT/'reports'
URLS=['https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv','https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv']
TARGETS=['median_house_value','MedHouseVal','target']

def main():
    for p in [DATA.parent, CHARTS, REPORTS]: p.mkdir(parents=True,exist_ok=True)
    sns.set_theme(style='whitegrid')
    if DATA.exists():
        try: df=pd.read_csv(DATA)
        except Exception: df=None
    else: df=None
    if df is None or df.empty:
        errors=[]
        for url in URLS:
            try:
                print('[INFO] Downloading dataset:',url)
                req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
                with urllib.request.urlopen(req,timeout=30) as r: DATA.write_bytes(r.read())
                df=pd.read_csv(DATA)
                if not df.empty: break
            except Exception as e:
                errors.append(f'{url}: {e}'); print('[WARN]',e)
        else:
            raise RuntimeError('Dataset download failed. Check internet or manually download a CSV from data/README.md and save it as data/housing.csv.\n'+'\n'.join(errors))
    target=next((c for c in TARGETS if c in df.columns),None)
    if target is None: raise ValueError(f'Cannot find target column. Available columns: {list(df.columns)}')
    before=len(df); df=df.drop_duplicates().copy(); df[target]=pd.to_numeric(df[target],errors='coerce'); df=df.dropna(subset=[target])
    print(f'[INFO] Dataset shape after cleanup: {df.shape}; target={target}')
    print('\nMissing values:\n',df.isna().sum().to_string())
    print('\nDescriptive statistics:\n',df.describe(include='all').transpose().to_string())
    # EDA charts
    plt.figure(figsize=(9,5)); sns.histplot(df[target].dropna(),kde=True); plt.title(f'Distribution of {target}'); plt.tight_layout(); plt.savefig(CHARTS/'target_distribution.png',dpi=160); plt.close()
    numeric=df.select_dtypes(include=np.number)
    if numeric.shape[1]>1:
        plt.figure(figsize=(11,8)); sns.heatmap(numeric.corr(),cmap='coolwarm',center=0); plt.title('Numeric Feature Correlation Heatmap'); plt.tight_layout(); plt.savefig(CHARTS/'correlation_heatmap.png',dpi=160); plt.close()
    missing=df.isna().sum(); missing=missing[missing>0]
    if len(missing):
        plt.figure(figsize=(9,4)); missing.plot(kind='bar'); plt.title('Missing Values by Column'); plt.tight_layout(); plt.savefig(CHARTS/'missing_values.png',dpi=160); plt.close()
    for col in [c for c in numeric.columns if c!=target][:6]:
        sample=df.sample(min(3000,len(df)),random_state=42)
        plt.figure(figsize=(7,4)); sns.scatterplot(data=sample,x=col,y=target,alpha=.4,s=18); plt.title(f'{target} vs {col}'); plt.tight_layout(); plt.savefig(CHARTS/f'target_vs_{col}.png',dpi=150); plt.close()
    X=df.drop(columns=[target]); y=df[target]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    nums=X_train.select_dtypes(include=np.number).columns.tolist(); cats=X_train.select_dtypes(exclude=np.number).columns.tolist()
    transformers=[]
    if nums: transformers.append(('numeric',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),nums))
    if cats: transformers.append(('categorical',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cats))
    results=[]; models={}; preds={}
    for name,est in [('Linear Regression',LinearRegression()),('Ridge',Ridge(alpha=1.0)),('Lasso',Lasso(alpha=.001,max_iter=20000))]:
        prep=ColumnTransformer(transformers=transformers,remainder='drop')
        model=Pipeline([('preprocessor',prep),('model',est)]); model.fit(X_train,y_train); p=model.predict(X_test)
        mse=float(mean_squared_error(y_test,p)); rmse=float(np.sqrt(mse)); r2=float(r2_score(y_test,p))
        results.append({'model':name,'MSE':mse,'RMSE':rmse,'R2':r2}); models[name]=model; preds[name]=p
        print(f'{name}: MSE={mse:,.3f} | RMSE={rmse:,.3f} | R²={r2:.4f}')
    metrics=pd.DataFrame(results).sort_values('RMSE'); metrics.to_csv(OUT/'metrics.csv',index=False)
    p=preds['Linear Regression']; residual=y_test.to_numpy()-p
    pd.DataFrame({'actual':y_test.to_numpy(),'predicted':p,'residual':residual}).to_csv(OUT/'predictions.csv',index=False)
    low=min(float(y_test.min()),float(np.min(p))); high=max(float(y_test.max()),float(np.max(p)))
    plt.figure(figsize=(7,6)); sns.scatterplot(x=y_test,y=p,alpha=.4,s=20); plt.plot([low,high],[low,high],'--'); plt.xlabel('Actual'); plt.ylabel('Predicted'); plt.title('Linear Regression: Actual vs Predicted'); plt.tight_layout(); plt.savefig(CHARTS/'actual_vs_predicted.png',dpi=160); plt.close()
    plt.figure(figsize=(8,5)); sns.scatterplot(x=p,y=residual,alpha=.4,s=20); plt.axhline(0,linestyle='--'); plt.xlabel('Predicted'); plt.ylabel('Residual (actual - predicted)'); plt.title('Residual Plot'); plt.tight_layout(); plt.savefig(CHARTS/'residual_plot.png',dpi=160); plt.close()
    lr=models['Linear Regression']; names=lr.named_steps['preprocessor'].get_feature_names_out(); coefs=lr.named_steps['model'].coef_
    coef=pd.DataFrame({'feature':names,'coefficient':coefs}); coef['absolute_coefficient']=coef.coefficient.abs(); coef.sort_values('absolute_coefficient',ascending=False).to_csv(OUT/'coefficients.csv',index=False)
    top=coef.sort_values('absolute_coefficient',ascending=False).head(15).sort_values('coefficient')
    plt.figure(figsize=(10,7)); sns.barplot(data=top,x='coefficient',y='feature'); plt.title('Largest Linear Regression Coefficients'); plt.tight_layout(); plt.savefig(CHARTS/'coefficient_analysis.png',dpi=160); plt.close()
    summary={'rows':int(len(df)),'columns':int(len(df.columns)),'target':target,'duplicate_rows_removed':int(before-len(df)),'missing_by_column':{str(k):int(v) for k,v in df.isna().sum().items()},'numeric_columns':nums,'categorical_columns':cats,'train_rows':len(X_train),'test_rows':len(X_test),'test_fraction':.2,'metrics':results,'dataset_note':'California district-level median house values; not individual home valuations.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    best=metrics.iloc[0]
    lines=['# House Price Prediction — Run Report','',f'- Rows after cleanup: **{len(df):,}**',f'- Columns: **{len(df.columns)}**',f'- Target: `{target}`',f'- Duplicate rows removed: **{before-len(df)}**','', '## Missing values before preprocessing']
    lines += [f'- `{c}`: {n}' for c,n in df.isna().sum().items()]
    lines += ['', '## Model results (20% test set)','','| Model | MSE | RMSE | R² |','|---|---:|---:|---:|']
    for row in metrics.to_dict(orient='records'): lines.append(f"| {row['model']} | {row['MSE']:,.3f} | {row['RMSE']:,.3f} | {row['R2']:.4f} |")
    lines += ['',f"Lowest RMSE in this run: **{best['model']}** (RMSE {best['RMSE']:,.3f}).",'These results are calculated by this run and are not a guarantee of future performance.','','## Interpretation checklist','- Describe the target distribution and important correlations.','- Does the actual-vs-predicted plot follow the diagonal?','- Are residuals randomly scattered or patterned?','- Compare RMSE and R² across Linear Regression, Ridge, and Lasso.','','## Limitations','- This dataset represents California census-block groups, not Indian homes or individual property listings.','- Linear Regression may underfit non-linear relationships.','- Use the metrics generated on your own computer; do not invent or copy results.']
    (REPORTS/'eda_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n[SUCCESS] Completed. See outputs/charts, outputs/metrics.csv, outputs/predictions.csv and outputs/reports/eda_report.md')

if __name__=='__main__':
    try: main()
    except Exception as e: print('\n[ERROR]',e,file=sys.stderr); raise
