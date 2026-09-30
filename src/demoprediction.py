import pandas as pd
import joblib
from pathlib import Path
from src.train import top_30_features

model_path = Path(__file__).resolve().parent.parent / "models" / "xgboost_baseline.joblib"

model=joblib.load(model_path)

def pred_features(top_30_features)-> list:

    top_30_model_features = top_30_features

    raw_data_features = [
    'ms_subclass', 'ms_zoning', 'lot_frontage', 'lot_area', 'street', 'alley', 
    'lot_shape', 'land_contour', 'utilities', 'lot_config', 'land_slope', 
    'neighborhood', 'condition_1', 'condition_2', 'bldg_type', 'house_style', 
    'overall_qual', 'overall_cond', 'year_built', 'year_remod/add', 'roof_style', 
    'roof_matl', 'exterior_1st', 'exterior_2nd', 'mas_vnr_type', 'mas_vnr_area', 
    'exter_qual', 'exter_cond', 'foundation', 'bsmt_qual', 'bsmt_cond', 
    'bsmt_exposure', 'bsmtfin_type_1', 'bsmtfin_sf_1', 'bsmtfin_type_2', 
    'bsmtfin_sf_2', 'bsmt_unf_sf', 'total_bsmt_sf', 'heating', 'heating_qc', 
    'central_air', 'electrical', '1st_flr_sf', '2nd_flr_sf', 'low_qual_fin_sf', 
    'gr_liv_area', 'bsmt_full_bath', 'bsmt_half_bath', 'full_bath', 'half_bath', 
    'bedroom_abvgr', 'kitchen_abvgr', 'kitchen_qual', 'totrms_abvgrd', 
    'functional', 'fireplaces', 'fireplace_qu', 'garage_type', 'garage_yr_blt', 
    'garage_finish', 'garage_cars', 'garage_area', 'garage_qual', 'garage_cond', 
    'paved_drive', 'wood_deck_sf', 'open_porch_sf', 'enclosed_porch', 
    '3ssn_porch', 'screen_porch', 'pool_area', 'pool_qc', 'fence', 'misc_feature', 
    'misc_val', 'mo_sold', 'yr_sold', 'sale_type', 'sale_condition'
]

    engineered_mappings = {
    "total_sq_ft": ["1st_flr_sf", "2nd_flr_sf", "total_bsmt_sf"],
    "total_bathrooms": ["full_bath", "half_bath", "bsmt_full_bath", "bsmt_half_bath"],
    "property_age": ["yr_sold", "year_built"]
}


    filtered_raw_features = []

    for item in top_30_model_features:

      if item.startswith("num__"):
          clean_name = item.replace("num__", "")
      elif item.startswith("cat__"):
          clean_name = item.replace("cat__", "")
          if "_" in clean_name:
              clean_name = clean_name.rsplit("_", 1)[0]
      else:
          clean_name = item
      if clean_name not in filtered_raw_features:
          if clean_name not in raw_data_features:
                for dependency in engineered_mappings[clean_name]:
                  if dependency not in filtered_raw_features:
                      filtered_raw_features.append(dependency)
          else:
              filtered_raw_features.append(clean_name)
    return filtered_raw_features


def main()-> None:
  single_house = {
    "gr_liv_area": 1656,       
    "1st_flr_sf": 1656,         
    "2nd_flr_sf": 0,            
    "low_qual_fin_sf": 0,       
    "total_bsmt_sf": 1080,      
    "bsmt_unf_sf": 441,         
    "bsmt_full_bath": 1,        
    "bsmt_half_bath": 0,        
    "full_bath": 1,             
    "half_bath": 0,             
    "bedroom_abvgr": 3,         
    "kitchen_abvgr": 1,         
    "totrms_abvgrd": 7,         
    "overall_qual": 6,          
    "overall_cond": 5,          
    "year_built": 1960,         
    "year_remod_add": 1960,     
    "yr_sold": 2010,            
    "mo_sold": 5,               
    "garage_cars": 2,           
    "garage_area": 528,         
    "fireplaces": 2,            
    "wood_deck_sf": 210,        
    "open_porch_sf": 62,        
    "enclosed_porch": 0,        
    "neighborhood": "NAmes",    
    "house_style": "1Story"
}
  single_house_1 = {
    "overall_qual": 6,
    "bsmt_qual": "TA",
    "exter_qual": "TA",
    "1st_flr_sf": 1656,
    "2nd_flr_sf": 0,
    "total_bsmt_sf": 1080,
    "central_air": "Y",
    "garage_cars": 2,
    "garage_cond": "TA",
    "ms_zoning": "RL",
    "kitchen_qual": "TA",
    "full_bath": 1,
    "half_bath": 0,
    "bsmt_full_bath": 1,
    "bsmt_half_bath": 0,
    "fireplaces": 2,
    "paved_drive": "P",
    "roof_style": "Hip",
    "year_remod/add": 1960,
    "gr_liv_area": 1656,
    "yr_sold": 2010,
    "year_built": 1960,
    "functional": "Typ",
    "fireplace_qu": "Gd",
    "neighborhood": "NAmes",
    "sale_condition": "Normal",
    "overall_cond": 5,
    "exter_cond": "TA"
}

  single_house_target=215000

  input_df = pd.DataFrame([single_house_1])
  predicted_price = model.predict(input_df)

  print("\n--- Model Inference Verification ---")
  print(f"Predicted Price : ${predicted_price[0]:,.2f}")
  print(f"Actual Price    : ${single_house_target:,.2f}")
  print(f"Difference      : ${abs(predicted_price[0] - single_house_target):,.2f}")
