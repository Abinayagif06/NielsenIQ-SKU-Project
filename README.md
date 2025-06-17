import pandas as pd
import re
import numpy as np
from tkinter import *
import tkinter.messagebox as tkmessagebox
from tkinter import filedialog
from customtkinter import*
from datetime import datetime
import os
import zipfile
import numpy as np
import warnings
from tkinter import messagebox
warnings.filterwarnings("ignore")
   

app = CTk()
app.title("SKU AUTOMATION")
set_appearance_mode("dark")
app.geometry("800x300")

SKUS = StringVar()
SKU_FILE_PATH =StringVar()
SKU_FILE_NAME = StringVar()
SKU_FILE = StringVar()

def SKU():
    SKU_filepath = filedialog.askopenfilename(filetypes=[("ZIP files", "*.zip")]) 
    SKUfilename = os.path.split(SKU_filepath)
    SKU_FILE_PATH.set(SKUfilename[0])  # File path only
    SKU_FILE_NAME.set(SKUfilename[1])  # File name only
    SKU_FILE.set(SKU_filepath)
    
def SKU_Smplifiy():
    with zipfile.ZipFile(SKU_FILE.get(), "r") as zip_ref:
         with zip_ref.open(zip_ref.namelist()[0]) as SKU:
                   child = pd.read_csv(SKU, skiprows=1,low_memory=False, dtype=str )


           # Step 2: Clean columns
                   if CATEGORY_button.get() == "PG DEODORANT":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)
                         
                        # Step 3: Split BP and Non-BP
                        bp_upcs = child[child['BARCODE'].str.contains('BP', na=False, case=False)].copy()
                        non_bp_upcs = child[~child['BARCODE'].str.contains('BP', na=False, case=False)].copy()
                         
                        # =============== BRAND CLEANING FUNCTION ===============
                        import re
                        def clean_brand_name(value):
                            if pd.isna(value):
                                return value
                            value = str(value)
                            # Remove nested parentheses and brackets
                            while re.search(r'\([^()]*\)', value):
                                value = re.sub(r'\([^()]*\)', '', value)
                            while re.search(r'\[[^\[\]]*\]', value):
                                value = re.sub(r'\[[^\[\]]*\]', '', value)
                            # Remove all colons
                            value = value.replace(':', '')
                            return re.sub(r'\s+', ' ', value).strip()
                         
                        # =============== NON-BP RULES ===============
                        def transform_non_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'NON_BP'
                         
                            def handle_subbrand(row):
                                val = str(row['PG SUBBRAND [-4574]']).upper()
                                if 'PRIVATE LABEL AO SB' in val:
                                    return 'PRIVATE LABEL'
                                elif 'NOT APPLICABLE' in val:
                                    return row['#US LOC BRAND [71177]']
                                elif any(x in val for x in ['AO SB', 'AO SUBBRAND']):
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG SUBBRAND [-4574]'])
                         
                            df['PG SUBBRAND [-4574]'] = df.apply(handle_subbrand, axis=1)
                         
                            # PG HEAD TYPE cleanup
                            df['PG VARIANT [-4576]'] = df['PG VARIANT [-4576]'].replace(['NOT STATED', 'AO SCENTS' ,'NOT COLLECTED'], np.nan)
                            
                            df['PG TYPE [-4575]'] = df['PG TYPE [-4575]'].replace(['AO TYPES','NOT COLLECTED'], np.nan)
                         
                            # BASE SIZE replacements
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].astype(str)
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)
                         
                            # MULTI PACK
                            df.loc[df['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = ''
                            df.loc[df['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False), '#US LOC IM MULTI CHAR [75626]'] += ' PK'
                         
                            # Aggregated value
                            aggregation_columns = [
                                'PG SUBBRAND [-4574]', 'PG VARIANT [-4576]', 'PG TYPE [-4575]',
                                'PG SEGMENT [-4573]', '#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]']
                            
                         
                            def aggregate_non_bp(row):
                                return ' '.join([
                                    str(row[col]).strip() for col in aggregation_columns
                                    if pd.notna(row[col]) and str(row[col]).strip()])
                         
                            df['AGGREGATED_VALUE'] = df.apply(aggregate_non_bp, axis=1)
                            return df
                         
                        # =============== BP RULES ===============
                        def transform_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'BP'
                         
                            def handle_brand(row):
                                val = str(row['PG BRAND [-4579]']).upper()
                                if 'PRIVATE LABEL AO' in val:
                                    return 'PRIVATE LABEL'
                                elif 'NOT APPLICABLE' in val:
                                    return row['#US LOC BRAND [71177]']
                                elif 'AO' in val:
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG BRAND [-4579]'])
                         
                            df['PG BRAND [-4579]'] = df.apply(handle_brand, axis=1)
                         
                            # Remove BP suffix from BARCODE
                            df['BARCODE'] = df['BARCODE'].str.extract(r'(\d{12})')
                         
                            # Aggregated value
                            def aggregate_bp(row):
                                parts = [row['PG BRAND [-4579]'], row['PG CATEGORY GIFT SET [-6191]']]
                                if row['PG BRAND [-4579]'] != 'PRIVATE LABEL':
                                    parts.append(row['BARCODE'])
                                return ' '.join([str(p).strip() for p in parts if pd.notna(p) and str(p).strip()])
                         
                            df['AGGREGATED_VALUE'] = df.apply(aggregate_bp, axis=1)
                            return df
                         
                        # Step 4: Apply transformations
                        transformed_non_bp = transform_non_bp(non_bp_upcs)
                        transformed_bp = transform_bp(bp_upcs)
                         
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG STOMACH REMEDIES":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        mask1 = child['PG SUBBRAND [-4329]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-4329]'] != 'PRIVATE LABEL AO SB')

                        # Create a mask for rows where PG SUBBRAND [-4329] is exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-4329]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])

                        # Combine both masks using OR (|)
                        mask = mask1 | mask2

                        # Replace with '#US LOC BRAND [71177]' while removing parentheses and content inside them
                        child.loc[mask, 'PG SUBBRAND [-4329]'] = (
                            child.loc[mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)  # Remove anything inside parentheses
                            .str.strip()
                        )


                        # If 'PG SUBBRAND [-4329]' is 'PRIVATE LABEL AO SB', replace it with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4329]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4329]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4329]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-4329]'] = child['#US LOC BRAND [71177]']

                        #To exclude the value in PG FLAVOR, if it has AO SB in it
                        child['PG FLAVOR [-4320]'] = child['PG FLAVOR [-4320]'].astype(str)  # Convert to string type
                        child.loc[child['PG FLAVOR [-4320]'].str.match(r'(?i)^(AO|REGULAR)', na=False), 'PG FLAVOR [-4320]'] = np.nan  # Replace values

                        child['PG FORM [-4321]'] = child['PG FORM [-4321]'].astype(str)  # Convert to string type
                        child.loc[child['PG FORM [-4321]'].str.contains(r'AO FORMS', na=False, case=False, regex=True), 'PG FORM [-4321]'] = np.nan  # Remove values

                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)  # Convert to string type

                        # Replace COUNT with CT
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)

                        # Replace FLUID OUNCE and OUNCE with OZ
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)
                        columns_to_combine = [
                            'PG SUBBRAND [-4329]', 'PG SEGMENT [-4325]', 'PG STRENGTH [-4327]', 
                            'PG FLAVOR [-4320]', 'PG FORM [-4321]', '#US LOC BASE SIZE [71802]', 
                            '#US LOC IM MULTI CHAR [75626]'
                        ]

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG FLOSS":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        mask1 = child['PG SUBBRAND [-5107]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-5107]'] != 'PRIVATE LABEL AO SB')

                        # Create a mask for rows where PG SUBBRAND [-4329] is exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-5107]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])

                        # Combine both masks using OR (|)
                        mask = mask1 | mask2

                        # Replace with '#US LOC BRAND [71177]' while removing parentheses and content inside them
                        child.loc[mask, 'PG SUBBRAND [-5107]'] = (
                            child.loc[mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)  # Remove anything inside parentheses
                            .str.strip()
                        )


                        # If 'PG SUBBRAND [-4329]' is 'PRIVATE LABEL AO SB', replace it with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-5107]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-5107]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-5107]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-5107]'] = child['#US LOC BRAND [71177]']

                        child['PG FLAVOR [-5098]'] = child['PG FLAVOR [-5098]'].astype(str)  # Convert to string type
                        child.loc[child['PG FLAVOR [-5098]'].str.match(r'(?i)^(AO|REGULAR)', na=False), 'PG FLAVOR [-5098]'] = np.nan  # Replace values

                        child['PG TYPE [-5108]'] = child['PG TYPE [-5108]'].astype(str)  # Convert to string type
                        child.loc[child['PG TYPE [-5108]'].str.match(r'(?i)^(AO TYPES)', na=False), 'PG TYPE [-5108]'] = np.nan
                        # Replace values

                        child = child[~child['#US LOC BRUSH HEAD SIZE [70964]'].isin(['NOT STATED', 'NOT APPLICABLE', 'NOT COLLECTED'])]

                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].astype(str) == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan

                        # Ensure 'BASE SIZE' column is a string
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)

                        # Replace 'COUNT' with 'CT' and 'OUNCE' with 'OZ'
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)

                        # Replace 'COUNT' with 'CT' and 'OUNCE' with 'OZ'
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'OUNCE', 'OZ', case=False, regex=True)

                        # Ensure 'IM MULTI' is treated as a string
                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)

                        # Add 'PK' only if 'IM MULTI' contains a number and is not empty
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != ''), '#US LOC IM MULTI CHAR [75626]'] += ' PK'


                        columns_to_combine = [
                            'PG SUBBRAND [-5107]', 'PG FORM [-5099]', 'PG FLAVOR [-5098]', 'PG TYPE [-5108]',
                            '#US LOC BRUSH HEAD SIZE [70964]','#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]'
                            ]

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]
                        child['AGGREGATED_DATA'] = child['AGGREGATED_DATA'].str.replace(r'\bnan\b', '', regex=True).str.strip()

                   elif CATEGORY_button.get() == "PG FABRIC CONDITIONER":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        mask1 = child['PG SUBBRAND [-4344]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-4344]'] != 'PRIVATE LABEL AO SB')

                        # Create a mask for rows where PG SUBBRAND [-4329] is exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-4344]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])

                        # Combine both masks using OR (|)
                        mask = mask1 | mask2

                        # Replace with '#US LOC BRAND [71177]' while removing parentheses and content inside them
                        child.loc[mask, 'PG SUBBRAND [-4344]'] = (
                            child.loc[mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)  # Remove anything inside parentheses
                            .str.strip()
                        )
                        child.loc[child['PG SCENT [-4354]'].str.upper() == 'NOT APPLICABLE', 'PG SCENT [-4354]'] = np.nan

                        child['PG PACKAGE TYPE [-4361]'] = child['PG PACKAGE TYPE [-4361]'].str.strip().str.upper()

                        # Remove "NOT APPLICABLE" and "AO PACKAGE TYPE"
                        child.loc[child['PG PACKAGE TYPE [-4361]'].isin(['NOT APPLICABLE', 'AO PACKAGE TYPES']), 'PG PACKAGE TYPE [-4361]'] = np.nan

                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)

                        # Replace 'COUNT' with 'CT' and 'OUNCE' with 'OZ'
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'OUNCE', 'OZ', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.strip().str.upper()

                        # Replace "FLUID OZ" with "OZ"
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'\bFLUID OZ\b', 'OZ', regex=True)
                        # Replace "OUNCE" with "OZ" (in case of variations like "OUNCE" alone)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'\bOUNCE\b', 'OZ', regex=True)

                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        columns_to_combine = [
                            'PG SUBBRAND [-4344]', 'PG SCENT [-4354]', 'PG SEGMENT [-4346]', 'PG PACKAGE TYPE [-4361]',
                            '#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]'
                            ]

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]
                        child['AGGREGATED_DATA'] = child['AGGREGATED_DATA'].str.replace(r'\bnan\b', '', regex=True).str.strip()

                   elif CATEGORY_button.get() == "PG LAWN CARE":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # === Replacement Logic ===

                        # Mask 1: ends with 'AO SB' but not 'PRIVATE LABEL AO SB'
                        mask1 = child['PG SUBBRAND [-36235]'].str.endswith('AO SB', na=False) & \
                                (child['PG SUBBRAND [-36235]'] != 'PRIVATE LABEL AO SB')

                        # Mask 2: exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-36235]'].isin(['AO SB', 'AO SUBBRAND'])

                        # Combine masks
                        mask = mask1 | mask2

                        # Replace with cleaned version of '#US LOC BRAND [71177]'
                        child.loc[mask, 'PG SUBBRAND [-36235]'] = (
                            child.loc[mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)
                            .str.strip()
                        )

                        # Replace 'PRIVATE LABEL AO SB' with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-36235]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-36235]'] = 'PRIVATE LABEL'

                        # Replace 'NOT APPLICABLE' with value from '#US LOC BRAND [71177]'
                        child.loc[child['PG SUBBRAND [-36235]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-36235]'] = (
                            child['#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)
                            .str.strip()
                        )

                        child['PG SYSTEM [-36358]'] = child['PG SYSTEM [-36358]'].apply(
                            lambda x: "" if isinstance(x, str) and x.strip().lower() == 'not applicable' else x
                        )

                        child['PG TYPE [-36355]'] = child['PG TYPE [-36355]'].str.strip().str.upper()

                        child.loc[child['PG TYPE [-36355]'].isin(['AO INGREDIENT', 'UNDEFINED','NOT APPLICABLE']), 'PG TYPE [-36355]'] = np.nan

                        child['PG FORM [-36377]'] = child['PG FORM [-36377]'].str.strip().str.upper()

                        child.loc[child['PG FORM [-36377]'].isin(['AO FORM', 'UNSCENTED']), 'PG FORM [-36377]'] =  np.nan

                        child['#US LOC PACKAGE GENERAL SHAPE [71807]'] = child['#US LOC PACKAGE GENERAL SHAPE [71807]'].str.strip().str.upper()

                        child.loc[child['#US LOC PACKAGE GENERAL SHAPE [71807]'].isin(['NOT AVAILABLE', 'NOT STATED','NOT COLLECTED','AO SUB FORMS']),'#US LOC PACKAGE GENERAL SHAPE [71807]'] =  np.nan

                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)

                        # Replace 'COUNT' with 'CT' and 'OUNCE' with 'OZ'
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)

                        # Replace 'COUNT' with 'CT' and 'OUNCE' with 'OZ'
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'OUNCE', 'OZ', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str).str.replace(r'FLUID OZ', 'OZ', case=False, regex=True)

                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[
                            child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False),
                            '#US LOC IM MULTI CHAR [75626]'
                        ] = child['#US LOC IM MULTI CHAR [75626]'] + ' PK'


                        columns_to_combine = [
                            'PG SUBBRAND [-36235]','PG SUBCATEGORY [-36388]','PG SYSTEM [-36358]','PG TYPE [-36355]','PG FORM [-36377]','#US LOC PACKAGE GENERAL SHAPE [71807]','#US LOC BASE SIZE [71802]','#US LOC IM MULTI CHAR [75626]'
                            ]
                        # Clean aggregation: drop both NaN and empty strings before joining
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(
                            lambda row: ' '.join(val for val in row if pd.notna(val) and val != ''), axis=1
                        )
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]
                        child['AGGREGATED_DATA'] = child['AGGREGATED_DATA'].str.replace(r'\s+', ' ', regex=True).str.strip()

                   elif CATEGORY_button.get() == "PG PEST CONTROL":
                        # Rename essential columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)
                        import re
                        def remove_nested_parentheses(text):
                            if pd.isna(text):
                                return text
                            result = []
                            level = 0
                            for char in text:
                                if char == '(':
                                    level += 1
                                elif char == ')':
                                    if level > 0:
                                        level -= 1
                                elif level == 0:
                                    result.append(char)
                            return re.sub(r'\s+', ' ', ''.join(result)).strip()

                        # AO replacements in PG SUBBRAND
                        sb_col = 'PG SUBBRAND [-25664]'
                        brand_col = '#US LOC BRAND [71177]'
                        child[sb_col] = child[sb_col].astype(str)

                        mask1 = child[sb_col].str.endswith('AO SB', na=False) & (child[sb_col] != 'PRIVATE LABEL AO SB')
                        mask2 = child[sb_col].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])
                        mask3 = child[sb_col].str.startswith('AO', na=False)
                        mask4 = child[sb_col].str.contains('AO', na=False)
                        mask = mask1 | mask2 | mask3 | mask4
                        child.loc[mask, sb_col] = child.loc[mask, brand_col].apply(remove_nested_parentheses)

                        child.loc[child[sb_col] == 'PRIVATE LABEL AO SB', sb_col] = 'PRIVATE LABEL'
                        child.loc[child[sb_col] == 'NOT APPLICABLE', sb_col] = child[brand_col]

                        # Clean scent, subform, system
                        child['PG SCENT [-25672]'] = child['PG SCENT [-25672]'].astype(str)
                        child.loc[child['PG SCENT [-25672]'].str.match(r'(?i)^UNSCENTED$', na=False), 'PG SCENT [-25672]'] = np.nan

                        child['PG SUBFORM [-25666]'] = child['PG SUBFORM [-25666]'].astype(str)
                        child.loc[child['PG SUBFORM [-25666]'].str.match(r'(?i)^AO SUBFORMS$', na=False), 'PG SUBFORM [-25666]'] = np.nan

                        child['PG SYSTEM [-38106]'] = child['PG SYSTEM [-38106]'].astype(str)
                        child.loc[child['PG SYSTEM [-38106]'].str.match(r'(?i)^NOT APPLICABLE$', na=False), 'PG SYSTEM [-38106]'] = np.nan

                        # PACKAGE TYPE & BASE SIZE logic
                        package_type_col = 'PG PACKAGE TYPE [-25665]'
                        base_size_col = '#US LOC BASE SIZE [71802]'

                        child[package_type_col] = child[package_type_col].astype(str)
                        child[base_size_col] = child[base_size_col].astype(str)

                        # Replace "AO COUNT" with BASE SIZE
                        child.loc[child[package_type_col].str.upper() == 'AO COUNT', package_type_col] = child[base_size_col]

                        # Count pattern like "1 COUNT", "2 COUNT"
                        count_pattern = r'^\d+\s*COUNT$'
                        mask_replace = (
                            child[package_type_col].str.match(count_pattern, na=False) &
                            child[base_size_col].str.match(count_pattern, na=False)
                        )
                        child.loc[mask_replace, package_type_col] = ""  # Leave PACKAGE TYPE blank so only BASE SIZE shows in aggregation

                        # Cleanup base size text
                        child[base_size_col] = child[base_size_col].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child[base_size_col] = child[base_size_col].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)
                        child[base_size_col] = child[base_size_col].str.replace(r'\s*CT$', ' CT', regex=True)

                        # Columns used in AGGREGATED_DATA
                        columns_to_combine = [
                            'PG SUBBRAND [-25664]', 'PG SEGMENT [-25673]', 
                            'PG TYPE [-25663]', 'PG SCENT [-25672]', 'PG SUBFORM [-25666]',
                            'PG SYSTEM [-38106]', package_type_col, base_size_col
                        ]

                        # Replace literal 'nan' and clean commas
                        child[columns_to_combine] = child[columns_to_combine].replace('nan', np.nan)
                        for col in columns_to_combine:
                            child[col] = child[col].str.replace(',', '', regex=False)

                        # Create AGGREGATED_DATA (PACKAGE TYPE comes before BASE SIZE)
                        def build_aggregated(row):
                            values = []
                            for col in columns_to_combine:
                                val = row[col]
                                if pd.notna(val) and str(val).strip():
                                    values.append(str(val).strip())
                            return ' '.join(values)

                        child['AGGREGATED_DATA'] = child.apply(build_aggregated, axis=1)

                        # Remove any unnamed columns (just for cleanliness)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]


                   elif CATEGORY_button.get() == "PG APP STYLING":
                        # Drop unnecessary columns
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Clean PG SUBBRAND
                        mask1 = child['PG SUBBRAND [-10959]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-10959]'] != 'PRIVATE LABEL AO SB')
                        mask2 = child['PG SUBBRAND [-10959]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])
                        mask = mask1 | mask2
                        child.loc[mask, 'PG SUBBRAND [-10959]'] = (
                            child.loc[mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)
                            .str.strip()
                        )
                        child.loc[child['PG SUBBRAND [-10959]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-10959]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-10959]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-10959]'] = child['#US LOC BRAND [71177]']

                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)

                        # Replace 'COUNT' with 'CT' and 'OUNCE' with 'OZ'
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)

                        # Replace 'COUNT' with 'CT' and 'OUNCE' with 'OZ'
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'OUNCE', 'OZ', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str).str.replace(r'FLUID OZ', 'OZ', case=False, regex=True)

                        # Clean SCENT
                        child['#US LOC SCENT [71469]'] = child['#US LOC SCENT [71469]'].astype(str)
                        child.loc[child['#US LOC SCENT [71469]'].str.match(r'(?i)^(NOT COLLECTED|NOT STATED)', na=False), '#US LOC SCENT [71469]'] = np.nan

                        # Clean MODEL NUMBER
                        model_number_col = '#US LOC MODEL NUMBER [71242]'
                        bp_model_col = '#US LOC BP CONCAT MODEL NUMBER [102741]'
                        commodity_group_col = '#US LOC COMMODITY GROUP [70565]'

                        child[model_number_col] = child[model_number_col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
                        child[bp_model_col] = child[bp_model_col].astype(str)
                        child[commodity_group_col] = child[commodity_group_col].astype(str)

                        # Remove NOT STATED/NOT COLLECTED
                        child.loc[child[model_number_col].str.upper().isin(['NOT STATED', 'NOT COLLECTED']), model_number_col] = np.nan

                        # Replace model number with BP CONCAT if it's a combo pack
                        combo_mask = child[commodity_group_col].str.contains(r'COMBO|COMBINATION PACK', case=False, na=False)
                        child.loc[combo_mask, model_number_col] = child.loc[combo_mask, bp_model_col]

                        # Clean MULTI PACK
                        multi_pack_col = '#US LOC IM MULTI CHAR [75626]'
                        child[multi_pack_col] = child[multi_pack_col].astype(str)
                        child.loc[child[multi_pack_col] == '1', multi_pack_col] = np.nan
                        child.loc[child[multi_pack_col].str.match(r'^\d+$', na=False) & (child[multi_pack_col] != '1'), multi_pack_col] += ' PK'

                        # Clean up spaces in all string columns
                        child = child.applymap(lambda x: ' '.join(x.strip().split()) if isinstance(x, str) else x)

                        # Combine into AGGREGATED_DATA
                        columns_to_combine = [
                            'PG SUBBRAND [-10959]', 'PG SEGMENT [-10947]', '#US LOC SCENT [71469]',
                            '#US LOC MODEL NUMBER [71242]', '#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]'
                        ]

                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(
                            lambda row: ' '.join(x.strip() for x in row.dropna().astype(str) if x.strip().lower() != 'nan'),
                            axis=1
                        )

                        # Remove unnamed columns if any
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG ADULT INCONTINENCE":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Clean PG SUBBRAND
                        mask1 = child['PG SUBBRAND [-4305]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-4305]'] != 'PRIVATE LABEL AO SB')
                        mask2 = child['PG SUBBRAND [-4305]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])
                        mask = mask1 | mask2

                        def clean_loc_brand(value):
                            if pd.isna(value):
                                return value

                            # Extract prefix before colon (e.g., 'ATTN')
                            prefix_match = re.match(r'^\s*([^:]+):', value)
                            prefix = prefix_match.group(1).strip() if prefix_match else ""

                            # Remove everything in parentheses (nested or not)
                            while re.search(r'\([^()]*\)', value):
                                value = re.sub(r'\([^()]*\)', '', value)

                            # Remove the original prefix to avoid duplication
                            value = re.sub(r'^\s*[^:]+:\s*', '', value)

                            # Combine prefix and cleaned value
                            cleaned = f"{prefix} {value}".strip() if prefix else value.strip()
                            return cleaned

                        # Apply cleaning to LOC BRAND
                        child['CLEANED_LOC_BRAND'] = child['#US LOC BRAND [71177]'].apply(clean_loc_brand)

                        # Apply only for relevant SUBBRAND values
                        child.loc[mask, 'PG SUBBRAND [-4305]'] = child.loc[mask, 'CLEANED_LOC_BRAND']

                        child.loc[child['PG SUBBRAND [-4305]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4305]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4305]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-4305]'] = child['CLEANED_LOC_BRAND']

                        child = child.apply(lambda col: col.str.replace(r'\(.*\)', '', regex=True).str.strip() if col.dtype == 'object' else col)

                        child['PG SEGMENT [-4307]'] = child['PG SEGMENT [-4307]'].astype(str)
                        child.loc[child['PG SEGMENT [-4307]'].str.match(r'(?i)^(AO AI SEGMENT)', na=False), 'PG SEGMENT [-4307]'] = np.nan

                        child['PG PHYSICAL NEEDS [-4314]'] = child['PG PHYSICAL NEEDS [-4314]'].astype(str)
                        child.loc[child['PG PHYSICAL NEEDS [-4314]'].str.match(r'(?i)^(NOT APPLICABLE)', na=False), 'PG PHYSICAL NEEDS [-4314]'] = np.nan

                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)

                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        columns_to_combine = [
                            'PG SUBBRAND [-4305]', 'PG SEGMENT [-4307]', 'PG AUDIENCE [-4309]', 'PG PHYSICAL NEEDS [-4314]',
                            'PG SIZE [-4317]', '#US LOC BASE SIZE [71802]','#US LOC IM MULTI CHAR [75626]'
                            ]

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]
                        child['AGGREGATED_DATA'] = child['AGGREGATED_DATA'].str.replace(r'\bnan\b', '', regex=True).str.strip()

                   elif CATEGORY_button.get() == "PG FACIAL TISSUE":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Define column name
                        subbrand_col = 'PG SUBBRAND [-4409]'

                        # 1. Replace AO-related values with #US LOC BRAND LOW OGRDS (cleaned)
                        mask_ao = (
                            child[subbrand_col].str.contains('AO SB', case=False, na=False) |
                            child[subbrand_col].str.upper().isin(['AO SUBBRANDS BASIC', 'AO SUBBRANDS LOTION'])
                        )

                        # Replace AO SUBBRANDS with #US LOC BRAND [71177] minus anything in parentheses
                        child.loc[mask_ao, subbrand_col] = (
                            child.loc[mask_ao, '#US LOC BRAND [71177]']
                            .str.replace(r'\(.*?\)', '', regex=True)  # Remove anything in parentheses
                            .str.strip()
                        )
                        # 2. Replace 'PRIVATE LABEL AO SB' with 'PRIVATE LABEL'
                        child.loc[
                            child[subbrand_col].str.upper() == 'PRIVATE LABEL AO SB',
                            subbrand_col
                        ] = 'PRIVATE LABEL'

                        child['PG PACKAGE TYPE [-4414]'] = child['PG PACKAGE TYPE [-4414]'].astype(str)
                        child.loc[child['PG PACKAGE TYPE [-4414]'].str.match(r'(?i)^(AO PACK TYPE)', na=False), 'PG PACKAGE TYPE [-4414]'] = np.nan

                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)


                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        columns_to_combine = [
                            'PG SUBBRAND [-4409]', 'PG PACKAGE TYPE [-4414]', 'PG PACK COUNT GROUP [-4417]', '#US LOC BASE SIZE [71802]','#US LOC IM MULTI CHAR [75626]'
                            ]

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]
                        child['AGGREGATED_DATA'] = child['AGGREGATED_DATA'].str.replace(r'\bnan\b', '', regex=True).str.strip()

                   elif CATEGORY_button.get() == "PG BATH TISSUE":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Define column name
                        mask1 = child['PG SUBBRAND [-4481]'].str.contains('AO SB', na=False) & (child['PG SUBBRAND [-4481]'] != 'PRIVATE LABEL AO SB')

                        # Create a mask for rows where PG SUBBRAND [-4329] is exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-4481]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])

                        mask3 = child['PG SUBBRAND [-4481]'].str.upper().str.startswith('AO SUBBRANDS', na=False)


                        # Combine both masks using OR (|)
                        mask = mask1 | mask2 | mask3

                        # Replace with '#US LOC BRAND [71177]' while removing parentheses and content inside them
                        child.loc[mask, 'PG SUBBRAND [-4481]'] = (
                            child.loc[mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)  # Remove anything inside parentheses
                            .str.strip()
                        )

                        # If 'PG SUBBRAND [-4329]' is 'PRIVATE LABEL AO SB', replace it with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4481]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4481]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4481]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-4481]'] = child['#US LOC BRAND [71177]']

                        subbrand_col = 'PG SUBBRAND [-4481]'

                        # Clean the subbrand values
                        import re

                        def clean_text(text):
                            if pd.isna(text):
                                return text
                            # Remove nested parentheses
                            while re.search(r'\([^()]*\)', text):
                                text = re.sub(r'\([^()]*\)', '', text)
                            # Remove nested square brackets
                            while re.search(r'\[[^\[\]]*\]', text):
                                text = re.sub(r'\[[^\[\]]*\]', '', text)
                            # Remove stray brackets, colons, and 'LTD'
                            text = re.sub(r'[\(\)\[\]:]', '', text)
                            text = re.sub(r'\bLTD\b', '', text, flags=re.IGNORECASE)
                            # Remove multiple spaces and strip
                            text = re.sub(r'\s{2,}', ' ', text).strip()
                            return text

                        child[subbrand_col] = child[subbrand_col].astype(str).apply(clean_text)

                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)  # Convert to string type

                        # Replace COUNT with CT
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)

                        # Replace FLUID OUNCE and OUNCE with OZ
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        columns_to_combine = [
                            'PG SUBBRAND [-4481]', 'PG SEGMENT [-4480]', 'PG SCENT [-4479]', 
                            '#US LOC BASE SIZE [71802]',  '#US LOC IM MULTI CHAR [75626]'
                        ]

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]


                   elif CATEGORY_button.get() == "PG SELF DIAGNOSTIC TISSUE":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # === Step 3: Define columns ===
                        pg_subbrand_col = 'PG SUBBRAND [-4383]'
                        pg_segment_col = 'PG SEGMENT [-4382]'
                        pg_time_col = 'PG TIME [-4385]'
                        pg_test_count_col = 'PG TEST COUNT [-4384]'
                        us_loc_brand_col = '#US LOC BRAND [71177]'
                        common_consumer_name_col = '#US LOC COMMON CONSUMER NAME [72078]'

                        agg1_segments = {'PREGNANCY TEST KITS', 'OVULATION TEST KITS'}
                        agg2_segments = {
                            'DRUGS OF ABUSE TEST KITS', 'HEALTH TEST KITS',
                            'OTHER WOMENS DIAG TEST KITS', 'FERTILITY TEST KITS', 'MENOPAUSE TEST KITS'
                        }
                        import re
                        # === Step 4: Helpers ===
                        def remove_nested_brackets(text):
                            if pd.isna(text):
                                return text
                            while re.search(r'\([^()]*\)', text):
                                text = re.sub(r'\([^()]*\)', '', text)
                            while re.search(r'\[[^\[\]]*\]', text):
                                text = re.sub(r'\[[^\[\]]*\]', '', text)
                            return re.sub(r'\s{2,}', ' ', text).strip()

                        def clean_value(val):
                            if pd.isna(val):
                                return ''
                            cleaned = str(val).strip().replace(':', '')
                            return re.sub(r'\s{2,}', ' ', cleaned)

                        # Normalize segment for matching
                        child[pg_segment_col] = child[pg_segment_col].str.upper().str.strip()

                        # === Step 5: Create masks for aggregation 1 and 2 ===
                        mask_agg1 = child[pg_segment_col].isin(agg1_segments)
                        mask_agg2 = child[pg_segment_col].isin(agg2_segments)

                        # === Step 6: Apply rules for Aggregation 1 ===
                        ao_sb_mask = mask_agg1 & child[pg_subbrand_col].str.contains('AO SB|AO SUBBRANDS', case=False, na=False)
                        child.loc[ao_sb_mask, pg_subbrand_col] = child.loc[ao_sb_mask, us_loc_brand_col].apply(remove_nested_brackets)

                        private_label_mask = mask_agg1 & (child[pg_subbrand_col].str.upper() == 'PRIVATE LABEL AO SB')
                        child.loc[private_label_mask, pg_subbrand_col] = 'PRIVATE LABEL'

                        child[pg_time_col] = child[pg_time_col].str.strip().replace(["NOT COLLECTED", "NOT STATED"], "")

                        # === Step 7: Apply rules for Aggregation 2 ===
                        ao_sb_mask2 = mask_agg2 & child[pg_subbrand_col].str.contains('AO SB|AO SUBBRANDS', case=False, na=False)
                        child.loc[ao_sb_mask2, pg_subbrand_col] = child.loc[ao_sb_mask2, us_loc_brand_col].apply(remove_nested_brackets)

                        private_label_mask2 = mask_agg2 & (child[pg_subbrand_col].str.upper() == 'PRIVATE LABEL AO SB')
                        child.loc[private_label_mask2, pg_subbrand_col] = 'PRIVATE LABEL'

                        # Fill COMMON CONSUMER NAME with SEGMENT where blank
                        child[common_consumer_name_col] = child[common_consumer_name_col].fillna('')
                        child.loc[mask_agg2 & (child[common_consumer_name_col].str.strip() == ''), common_consumer_name_col] = child[pg_segment_col]

                        # === Step 8: Generate AGGREGATED_SKU ===
                        def generate_aggregated_sku(row):
                            segment = row[pg_segment_col]
                            if segment in agg1_segments:
                                parts = [row[pg_subbrand_col], row[pg_segment_col], row[pg_time_col], row[pg_test_count_col]]
                            elif segment in agg2_segments:
                                parts = [row[pg_subbrand_col], row[common_consumer_name_col], row[pg_time_col], row[pg_test_count_col]]
                            else:
                                return ''
                            cleaned_parts = [clean_value(x) for x in parts if pd.notna(x) and str(x).strip().lower() != 'nan']
                            return re.sub(r'\s{2,}', ' ', ' '.join(cleaned_parts)).strip()

                        child['AGGREGATED_SKU'] = child.apply(generate_aggregated_sku, axis=1)

                        # Final safety cleanup: remove any unnamed columns and multiple spaces
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]
                        child['AGGREGATED_SKU'] = child['AGGREGATED_SKU'].str.replace(r'\s{2,}', ' ', regex=True)


                   elif CATEGORY_button.get() == "PG BLADES AND RAZORS":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Step 3: Split BP and Non-BP
                        bp_upcs = child[child['BARCODE'].str.contains('BP', na=False, case=False)].copy()
                        non_bp_upcs = child[~child['BARCODE'].str.contains('BP', na=False, case=False)].copy()

                        # =============== BRAND CLEANING FUNCTION ===============
                        import re
                        def clean_brand_name(value):
                            if pd.isna(value):
                                return value
                            value = str(value)
                            # Remove nested parentheses and brackets
                            while re.search(r'\([^()]*\)', value):
                                value = re.sub(r'\([^()]*\)', '', value)
                            while re.search(r'\[[^\[\]]*\]', value):
                                value = re.sub(r'\[[^\[\]]*\]', '', value)
                            # Remove anything before colon
                            value = re.sub(r'^\s*[^:]+:\s*', '', value)
                            return re.sub(r'\s+', ' ', value).strip()

                        # =============== NON-BP RULES ===============
                        def transform_non_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'NON_BP'

                            def handle_subbrand(row):
                                val = str(row['PG SUBBRAND [-10837]']).upper()
                                if 'PRIVATE LABEL AO SB' in val:
                                    return 'PRIVATE LABEL'
                                elif 'NOT APPLICABLE' in val:
                                    return row['#US LOC BRAND [71177]']
                                elif any(x in val for x in ['AO SB', 'AO SUBBRAND', 'AO']):
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG SUBBRAND [-10837]'])

                            df['PG SUBBRAND [-10837]'] = df.apply(handle_subbrand, axis=1)

                            # PG HEAD TYPE cleanup
                            df['PG HEAD TYPE [-11198]'] = df['PG HEAD TYPE [-11198]'].replace('NOT IDENTIFIED', np.nan)

                            # BASE SIZE replacements
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].replace({
                                'COUNT': 'CT', 'OUNCE': 'OZ', 'FLUID OUNCE': 'OZ'
                            }, regex=True)

                            # MULTI PACK
                            df.loc[df['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = ''
                            df.loc[df['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                            # Aggregated value
                            aggregation_columns = [
                                'PG SUBBRAND [-10837]', 'PG HEAD TYPE [-11198]', 'PG HEAD SIZE [-10840]',
                                'PG SEGMENT [-10826]', '#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]'
                            ]

                            def aggregate_non_bp(row):
                                return ' '.join([
                                    str(row[col]).strip() for col in aggregation_columns
                                    if pd.notna(row[col]) and str(row[col]).strip()
                                ])

                            df['AGGREGATED_VALUE'] = df.apply(aggregate_non_bp, axis=1)
                            return df

                        # =============== BP RULES ===============
                        def transform_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'BP'

                            def handle_brand(row):
                                val = str(row['PG BRAND [-10833]']).upper()
                                if 'PRIVATE LABEL AO' in val:
                                    return 'PRIVATE LABEL'
                                elif 'NOT APPLICABLE' in val:
                                    return row['#US LOC BRAND [71177]']
                                elif 'AO' in val:
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG BRAND [-10833]'])

                            df['PG BRAND [-10833]'] = df.apply(handle_brand, axis=1)

                            # Remove BP suffix from BARCODE
                            df['BARCODE'] = df['BARCODE'].str.extract(r'(\d{12})')

                            # Aggregated value
                            def aggregate_bp(row):
                                parts = [row['PG BRAND [-10833]'], row['PG CATEGORY GIFT SET [-11196]']]
                                if row['PG BRAND [-10833]'] != 'PRIVATE LABEL':
                                    parts.append(row['BARCODE'])
                                return ' '.join([str(p).strip() for p in parts if pd.notna(p) and str(p).strip()])
                            df['AGGREGATED_VALUE'] = df.apply(aggregate_bp, axis=1)
                            return df

                        # Step 4: Apply transformations
                        transformed_non_bp = transform_non_bp(non_bp_upcs)
                        transformed_bp = transform_bp(bp_upcs)

                   elif CATEGORY_button.get() == "PG BABY AND KID WIPES":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)


                        # Define function to clean #US LOC BRAND by removing nested brackets and content after colons
                        import re
                        def remove_nested_parentheses(text):
                            if pd.isna(text):
                                return text

                            result = []
                            level = 0
                            for char in text:
                                if char == '(':
                                    level += 1
                                elif char == ')':
                                    if level > 0:
                                        level -= 1
                                elif level == 0:
                                    result.append(char)

                            return ''.join(result).strip()
                            return re.sub(r'\s+', ' ', ''.join(result)).strip()


                        mask1 = child['PG SUBBRAND [-4293]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-4293]'] != 'PRIVATE LABEL AO SB')

                        # Create a mask for rows where PG SUBBRAND [-4329] is exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-4293]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])

                        mask3 = child['PG SUBBRAND [-4293]'].str.startswith('AO', na=False)

                        # Combine both masks using OR (|)
                        mask = mask1 | mask2 | mask3

                        # Replace with '#US LOC BRAND [71177]' while removing parentheses and content inside them
                        child.loc[mask, 'PG SUBBRAND [-4293]'] = child.loc[mask, '#US LOC BRAND [71177]'].apply(remove_nested_parentheses)


                        # If 'PG SUBBRAND [-4329]' is 'PRIVATE LABEL AO SB', replace it with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4293]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4293]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4293]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-4293]'] = child['#US LOC BRAND [71177]']

                        child['#US LOC PACKAGE GENERAL SHAPE [71807]'] = child['#US LOC PACKAGE GENERAL SHAPE [71807]'].astype(str)  # Convert to string type
                        child.loc[child['#US LOC PACKAGE GENERAL SHAPE [71807]'].str.match(r'(?i)^(NOT APPLICABLE|NOT COLLECTED|NOT STATED)', na=False), '#US LOC PACKAGE GENERAL SHAPE [71807]'] = np.nan 

                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)  # Convert to string type

                        # Replace COUNT with CT
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)

                        # Replace FLUID OUNCE and OUNCE with OZ
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)
                        columns_to_combine = [
                            'PG SUBBRAND [-4293]', 'PG PRICE SEGMENT [-4299]', 'PG SEGMENT [-4290]', 
                            'PG SCENT [-4289]', 'PG PACKAGE TYPE [-4296]','#US LOC PACKAGE GENERAL SHAPE [71807]','#US LOC BASE SIZE [71802]', 
                            '#US LOC IM MULTI CHAR [75626]'
                        ]

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]


                   elif CATEGORY_button.get() == "PG DENTURE CARE":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Clean PG SUBBRAND for AO SB and AO SUBBRANDS
                        mask1 = child['PG SUBBRAND [-4878]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-4878]'] != 'PRIVATE LABEL AO SB')
                        mask2 = child['PG SUBBRAND [-4878]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])
                        mask = mask1 | mask2
                        child.loc[mask, 'PG SUBBRAND [-4878]'] = (
                            child.loc[mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)
                            .str.strip()
                        )

                        # Clean PG FLAVOR
                        child['PG FLAVOR [-4881]'] = child['PG FLAVOR [-4881]'].astype(str)
                        child.loc[child['PG FLAVOR [-4881]'].str.match(r'(?i)^(UNFLAVORED)', na=False), 'PG FLAVOR [-4881]'] = np.nan

                        # Clean PG FORM — only applying your specified logic
                        child['PG FORM [-4875]'] = child['PG FORM [-4875]'].astype(str).str.strip().str.upper()
                        child['#US LOC FORM [71782]'] = child['#US LOC FORM [71782]'].astype(str)

                        # Replace "AO FORMS" specifically with #US LOC FORM
                        mask_ao_forms = child['PG FORM [-4875]'] == 'AO FORMS'
                        child.loc[mask_ao_forms, 'PG FORM [-4875]'] = (
                            child.loc[mask_ao_forms, '#US LOC FORM [71782]']
                            .str.replace(r"\(.*?\)", "", regex=True)
                            .str.strip()
                        )

                        # Also handle 'PRIVATE LABEL AO FORM'
                        child.loc[child['PG FORM [-4875]'] == 'PRIVATE LABEL AO FORM', 'PG FORM [-4875]'] = 'PRIVATE LABEL'

                        # Clean MULTI PACK
                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan
                        child.loc[
                            child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) &
                            (child['#US LOC IM MULTI CHAR [75626]'] != '1'),
                            '#US LOC IM MULTI CHAR [75626]'
                        ] += ' PK'

                        # Clean BASE SIZE
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                        # Function to clean out 'nan' (as string) from selected columns
                        def remove_nan_text(df, columns):
                            for col in columns:
                                df[col] = (
                                    df[col]
                                    .astype(str)
                                    .str.replace(r'\bnan\b', '', case=False, regex=True)
                                    .str.replace(r'\s+', ' ', regex=True)
                                    .str.strip()
                                )
                            return df

                        # Columns to clean and combine
                        columns_to_combine = [
                            'PG SUBBRAND [-4878]', 'PG FLAVOR [-4881]', 'PG STRENGTH [-4884]', 'PG FORM [-4875]',
                            '#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]'
                        ]

                        # Clean before aggregation
                        child = remove_nan_text(child, columns_to_combine)

                        # Combine into AGGREGATED_DATA
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.replace('nan','')

                        # Drop unnamed columns
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG FACIAL SKINCARE":
                                            
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Step 3: Split BP and Non-BP
                        bp_upcs = child[child['BARCODE'].str.contains('BP', na=False, case=False)].copy()
                        non_bp_upcs = child[~child['BARCODE'].str.contains('BP', na=False, case=False)].copy()

                        # =============== BRAND CLEANING FUNCTION ===============
                        import re
                        def clean_brand_name(value):
                            if pd.isna(value):
                                return value
                            value = str(value)
                            import re
                            # Remove nested parentheses and brackets
                            pattern = r'\([^()]*\)|\[[^\[\]]*\]'
                            while re.search(pattern, value):
                                value = re.sub(pattern, '', value)

                            value = value.replace(':', '').replace(',', '')
                            value = re.sub(r'\s+', ' ', value).strip()
                            return value

                        # =============== NON-BP RULES ===============
                        def transform_non_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'NON_BP'

                            def handle_subbrand(row):
                                val = str(row['PG SUBBRAND [-5580]']).upper()
                                if 'PRIVATE LABEL AO' in val:
                                    return 'PRIVATE LABEL'
                                elif 'AO' in val:
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG SUBBRAND [-5580]'])

                            df['PG SUBBRAND [-5580]'] = df.apply(handle_subbrand, axis=1)

                            df['PG INGREDIENT [-5330]'] = df['PG INGREDIENT [-5330]'].replace(['NO SPF', 'AO SPF'], np.nan)
                            df['PG FORM [-5325]'] = df['PG FORM [-5325]'].replace('AO FORMS', np.nan)

                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].astype(str)
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                            df.loc[df['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = ''
                            df.loc[df['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                            aggregation_columns = [
                                'PG SUBBRAND [-5580]', 'PG DAYPART [-5298]', 'PG USE [-5581]', 'PG SCENT [-5578]',
                                'PG INGREDIENT [-5330]', 'PG FORM [-5325]', '#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]'
                            ]

                            def aggregate_non_bp(row):
                                return ' '.join([
                                    str(row[col]).strip().replace(',', '') for col in aggregation_columns
                                    if pd.notna(row[col]) and str(row[col]).strip()
                                ])

                            df['AGGREGATED_VALUE'] = df.apply(aggregate_non_bp, axis=1)
                            return df

                        # =============== BP RULES ===============
                        def transform_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'BP'

                            def handle_brand(row):
                                val = str(row['PG BRAND [-5585]']).upper()
                                if 'PRIVATE LABEL AO' in val:
                                    return 'PRIVATE LABEL'
                                elif 'AO' in val:
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG BRAND [-5585]'])

                            df['PG BRAND [-5585]'] = df.apply(handle_brand, axis=1)

                            df['BARCODE'] = df['BARCODE'].str.extract(r'(\d{12})')
                            df['PG CATEGORY GIFT SET [-6179]'] = df['PG CATEGORY GIFT SET [-6179]'].replace('NOT APPLICABLE', np.nan)

                            def aggregate_bp(row):
                                parts = [row['PG BRAND [-5585]']]
                                if pd.notna(row['PG CATEGORY GIFT SET [-6179]']) and str(row['PG CATEGORY GIFT SET [-6179]']).strip():
                                    parts.append(row['PG CATEGORY GIFT SET [-6179]'])
                                if row['PG BRAND [-5585]'] != 'PRIVATE LABEL':
                                    parts.append(row['BARCODE'])
                                return ' '.join([str(p).strip().replace(',', '') for p in parts if pd.notna(p) and str(p).strip()])

                            df['AGGREGATED_VALUE'] = df.apply(aggregate_bp, axis=1)
                            return df

                        # Step 4: Apply transformations
                        transformed_non_bp = transform_non_bp(non_bp_upcs)
                        transformed_bp = transform_bp(bp_upcs)

                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG DIAPERS":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        mask1 = child['PG SUBBRAND [-3843]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-3843]'] != 'PRIVATE LABEL AO SB')

                        # Create a mask for rows where PG SUBBRAND [-4329] is exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-3843]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])

                        # Combine both masks using OR (|)
                        mask = mask1 | mask2

                        # Replace with '#US LOC BRAND [71177]' while removing parentheses and content inside them
                        child.loc[mask, 'PG SUBBRAND [-3843]'] = (
                            child.loc[mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)  # Remove anything inside parentheses
                            .str.strip()
                        )


                        # If 'PG SUBBRAND [-4329]' is 'PRIVATE LABEL AO SB', replace it with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-3843]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-3843]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-3843]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-3843]'] = child['#US LOC BRAND [71177]']

                        #To exclude the value in PG FLAVOR, if it has AO SB in it
                        child['PG DIAPER SIZE [-4188]'] = child['PG DIAPER SIZE [-4188]'].astype(str)  # Convert to string type
                        child.loc[child['PG DIAPER SIZE [-4188]'].str.match(r'(?i)^(AO DIAPER SIZES)', na=False), 'PG DIAPER SIZE [-4188]'] = np.nan  # Replace values

                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)  # Convert to string type

                        # Replace COUNT with CT
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)

                        # Replace FLUID OUNCE and OUNCE with OZ
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)
                        columns_to_combine = [
                            'PG SUBBRAND [-3843]', 'PG GENDER [-3836]', 'PG PRICE SEGMENT [-3844]', 
                            'PG SEGMENT [-3841]', 'PG DIAPER SIZE [-4188]', '#US LOC BASE SIZE [71802]', 
                            '#US LOC IM MULTI CHAR [75626]'
                        ]

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG DIGESTIVE HEALTH":
                        import re

                        # Function to clean brand name by removing content inside parentheses/brackets
                        def clean_brand_name(value):
                            if pd.isna(value) or str(value).strip().lower() in ['nan', '']:
                                return None
                            value = str(value)
                            pattern = r'\([^()]*\)|\[[^\[\]]*\]'
                            while re.search(pattern, value):
                                value = re.sub(pattern, '', value)
                            value = re.sub(r'[:,]', '', value)
                            value = re.sub(r'\s+', ' ', value).strip()
                            return value if value.lower() != 'nan' and value != '' else None

                        # Drop unnecessary columns
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Clean #US LOC BRAND for future use
                        child['#US LOC BRAND [71177]'] = child['#US LOC BRAND [71177]'].apply(clean_brand_name)

                        # Convert PG SUBBRAND to string before processing
                        child['PG SUBBRAND [-4241]'] = child['PG SUBBRAND [-4241]'].astype(str)

                        # Create masks to identify AO values
                        mask1 = child['PG SUBBRAND [-4241]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-4241]'] != 'PRIVATE LABEL AO SB')
                        mask2 = child['PG SUBBRAND [-4241]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])
                        mask = mask1 | mask2

                        # Replace AO subbrand values with cleaned #US LOC BRAND
                        child.loc[mask, 'PG SUBBRAND [-4241]'] = child.loc[mask, '#US LOC BRAND [71177]']

                        # Replace 'PRIVATE LABEL AO SB' with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4241]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4241]'] = 'PRIVATE LABEL'

                        # Replace 'NOT APPLICABLE' with cleaned #US LOC BRAND
                        child.loc[child['PG SUBBRAND [-4241]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-4241]'] = child['#US LOC BRAND [71177]']

                        # Clean PG SUBFLAVOR
                        child['PG SUBFLAVOR [-42571]'] = child['PG SUBFLAVOR [-42571]'].astype(str)
                        child.loc[
                            child['PG SUBFLAVOR [-42571]'].str.match(r'(?i)^(AO FLAVORS|SUBFLAVOR UNDEFINED|UNFLAVORED)', na=False),
                            'PG SUBFLAVOR [-42571]'
                        ] = np.nan

                        # Clean PG SUGAR CONTENT
                        child['PG SUGAR CONTENT [-4249]'] = child['PG SUGAR CONTENT [-4249]'].astype(str)
                        child.loc[
                            child['PG SUGAR CONTENT [-4249]'].str.match(r'(?i)^SUGAR CONTENT UNDEFINED', na=False),
                            'PG SUGAR CONTENT [-4249]'
                        ] = np.nan

                        # Clean PG BENEFIT
                        child['PG BENEFIT [-4236]'] = child['PG BENEFIT [-4236]'].astype(str)
                        child.loc[
                            child['PG BENEFIT [-4236]'].str.match(r'(?i)^BENEFIT UNDEFINED', na=False),
                            'PG BENEFIT [-4236]'
                        ] = np.nan

                        # Clean PG SUBFORM
                        child['PG SUBFORM [-42570]'] = child['PG SUBFORM [-42570]'].astype(str)
                        child.loc[
                            child['PG SUBFORM [-42570]'].str.match(r'(?i)^AO SUBFORMS', na=False),
                            'PG SUBFORM [-42570]'
                        ] = np.nan

                        # Handle MULTIPACK values
                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan
                        child.loc[
                            child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'),
                            '#US LOC IM MULTI CHAR [75626]'
                        ] = child['#US LOC IM MULTI CHAR [75626]'] + ' PK'

                        # Clean BASE SIZE
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                        # Columns to combine
                        columns_to_combine = [
                            'PG SUBBRAND [-4241]',
                            'PG SUBFLAVOR [-42571]',
                            'PG SUGAR CONTENT [-4249]',
                            'PG SEGMENT [-4238]',
                            'PG SUBFORM [-42570]',
                            '#US LOC BASE SIZE [71802]',
                            '#US LOC IM MULTI CHAR [75626]'
                        ]

                        # Create AGGREGATED_DATA
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(
                            lambda row: ' '.join([str(x).strip() for x in row if pd.notna(x) and str(x).strip() != '']),
                            axis=1
                        )

                        # Drop unnamed columns
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG DISH CARE":

                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        mask1 = child['PG SUBBRAND [-4262]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-4262]'] != 'PRIVATE LABEL AO SB')

                        # Create a mask for rows where PG SUBBRAND [-4329] is exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-4262]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])

                        # Combine both masks using OR (|)
                        mask = mask1 | mask2

                        # Replace with '#US LOC BRAND [71177]' while removing parentheses and content inside them
                        child.loc[mask, 'PG SUBBRAND [-4262]'] = (
                            child.loc[mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)  # Remove anything inside parentheses
                            .str.strip()
                        )


                        # If 'PG SUBBRAND [-4329]' is 'PRIVATE LABEL AO SB', replace it with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4262]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4262]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4262]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-4262]'] = child['#US LOC BRAND [71177]']

                        #To exclude the value in PG FLAVOR, if it has AO SB in it
                        child['PG SCENT GROUP [-4260]'] = child['PG SCENT GROUP [-4260]'].astype(str)  # Convert to string type
                        child.loc[child['PG SCENT GROUP [-4260]'].str.match(r'(?i)^(NOT STATED|NOT COLLECTED|N/A)', na=False), 'PG SCENT GROUP [-4260]'] = np.nan  # Replace values

                        child['PG FORM [-4259]'] = child['PG FORM [-4259]'].astype(str)  # Convert to string type
                        child.loc[child['PG FORM [-4259]'].str.match(r'(?i)^(AO FORMS)', na=False), 'PG FORM [-4259]'] = np.nan 

                        child['PG VARIETY [-4396]'] = child['PG VARIETY [-4396]'].astype(str)  # Convert to string type
                        child.loc[child['PG VARIETY [-4396]'].str.match(r'(?i)^(AO VARIETIES)', na=False), 'PG VARIETY [-4396]'] = np.nan 


                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)  # Convert to string type

                        # Replace COUNT with CT
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)

                        # Replace FLUID OUNCE and OUNCE with OZ
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)
                        columns_to_combine = [
                            'PG SUBBRAND [-4262]','PG SCENT GROUP [-4260]','PG SEGMENT [-4263]','PG FORM [-4259]','PG VARIETY [-4396]','#US LOC BASE SIZE [71802]', 
                            '#US LOC IM MULTI CHAR [75626]'
                        ]

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG LAUNDRY CARE":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Handle PG SUBBRAND AO replacements
                        child['PG SUBBRAND [-4447]'] = child['PG SUBBRAND [-4447]'].astype(str)
                        child['#US LOC BRAND [71177]'] = child['#US LOC BRAND [71177]'].astype(str)

                        # Identify rows where PG SUBBRAND contains "AO"
                        ao_mask = child['PG SUBBRAND [-4447]'].str.contains(r'\bAO\b', case=False, na=False)

                        # Clean #US LOC BRAND by removing bracketed content (including parentheses)
                        child.loc[ao_mask, 'PG SUBBRAND [-4447]'] = (
                            child.loc[ao_mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)|\[.*?\]", "", regex=True)
                            .str.strip()
                        )

                        # Replace PRIVATE LABEL AO SB with PRIVATE LABEL
                        child.loc[child['PG SUBBRAND [-4447]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4447]'] = 'PRIVATE LABEL'

                        # Replace NOT APPLICABLE with cleaned #US LOC BRAND
                        not_applicable_mask = child['PG SUBBRAND [-4447]'] == 'NOT APPLICABLE'
                        child.loc[not_applicable_mask, 'PG SUBBRAND [-4447]'] = (
                            child.loc[not_applicable_mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)|\[.*?\]", "", regex=True)
                            .str.strip()
                        )

                        # Handle PG SHAPE cleanup
                        child['#US LOC PACKAGE GENERAL SHAPE [71807]'] = child['#US LOC PACKAGE GENERAL SHAPE [71807]'].astype(str)
                        child.loc[child['#US LOC PACKAGE GENERAL SHAPE [71807]'].str.match(r'(?i)^(NOT STATED|NOT COLLECTED)', na=False), '#US LOC PACKAGE GENERAL SHAPE [71807]'] = np.nan

                        # MULTI PACK formatting
                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        # BASE SIZE formatting
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                        # Combine for aggregation
                        columns_to_combine = [
                            'PG SUBBRAND [-4447]', 'PG SCENT [-4446]', 'PG SEGMENT [-4462]',
                            'PG MACHINE TYPE [-4460]', '#US LOC PACKAGE GENERAL SHAPE [71807]',
                            '#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]'
                        ]

                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG ORAL":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Ensure string types
                        child['PG SUBBRAND [-4520]'] = child['PG SUBBRAND [-4520]'].astype(str)
                        child['#US LOC BRAND [71177]'] = child['#US LOC BRAND [71177]'].astype(str)

                        # Replace exact "PRIVATE LABEL AO SB" with "PRIVATE LABEL"
                        child.loc[child['PG SUBBRAND [-4520]'].str.strip().str.upper() == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4520]'] = 'PRIVATE LABEL'

                        # Replace "NOT APPLICABLE" with cleaned brand
                        not_applicable_mask = child['PG SUBBRAND [-4520]'].str.strip().str.upper() == 'NOT APPLICABLE'
                        child.loc[not_applicable_mask, 'PG SUBBRAND [-4520]'] = (
                            child.loc[not_applicable_mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)|\[.*?\]", "", regex=True)
                            .str.strip()
                        )

                        # Replace remaining values containing "AO" with cleaned #US LOC BRAND
                        ao_mask = child['PG SUBBRAND [-4520]'].str.contains(r'\bAO\b', case=False, na=False)
                        child.loc[ao_mask, 'PG SUBBRAND [-4520]'] = (
                            child.loc[ao_mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)|\[.*?\]", "", regex=True)
                            .str.strip()
                        )

                        # Clean and fix flavor column
                        child['#US LOC FLAVOR [71057]'] = child['#US LOC FLAVOR [71057]'].astype(str)
                        child.loc[child['#US LOC FLAVOR [71057]'].str.strip().str.upper().isin(['AO FLAVORS', 'NOT STATED','NOT COLLECTED']), '#US LOC FLAVOR [71057]'] = np.nan

                        # Replace missing flavor with BP CONCAT FLAVOR
                        child['#US LOC BP CONCAT FLAVOR [96356]'] = child['#US LOC BP CONCAT FLAVOR [96356]'].astype(str)
                        flavor_missing = child['#US LOC FLAVOR [71057]'].isna() | (child['#US LOC FLAVOR [71057]'].str.strip().str.lower() == 'nan')
                        child.loc[flavor_missing, '#US LOC FLAVOR [71057]'] = child.loc[flavor_missing, '#US LOC BP CONCAT FLAVOR [96356]']

                        # Clean common consumer name
                        child['#US LOC COMMON CONSUMER NAME [72078]'] = child['#US LOC COMMON CONSUMER NAME [72078]'].astype(str)
                        child.loc[child['#US LOC COMMON CONSUMER NAME [72078]'].str.match(r'(?i)^AO TYPES$', na=False), '#US LOC COMMON CONSUMER NAME [72078]'] = np.nan

                        # Clean group benefit
                        child['PG GROUP BENEFIT [-4107]'] = child['PG GROUP BENEFIT [-4107]'].astype(str)
                        child.loc[child['PG GROUP BENEFIT [-4107]'].str.match(r'(?i)^AO GROUP BENEFITS$', na=False), 'PG GROUP BENEFIT [-4107]'] = np.nan

                        # MULTI PACK formatting
                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan
                        child.loc[
                            child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'),
                            '#US LOC IM MULTI CHAR [75626]'
                        ] += ' PK'

                        # BASE SIZE formatting
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                        # Combine for AGGREGATED_DATA, excluding real NaNs and literal "nan"
                        columns_to_combine = [
                            'PG SUBBRAND [-4520]',
                            '#US LOC FLAVOR [71057]',
                            '#US LOC COMMON CONSUMER NAME [72078]',
                            'PG GROUP BENEFIT [-4107]',
                            '#US LOC BASE SIZE [71802]',
                            '#US LOC IM MULTI CHAR [75626]'
                        ]

                        def clean_join(row):
                            return ' '.join(val for val in row if pd.notna(val) and str(val).strip().lower() != 'nan')

                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(clean_join, axis=1)

                        # Drop Unnamed columns
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG PAPER TOWELS":
                        import re

                        # Clean brand name by removing content in brackets, colons, commas, and extra spaces
                        def clean_brand_name(value):
                            if pd.isna(value) or value == 'nan':
                                return None
                            value = str(value)
                            # Remove all content inside parentheses and square brackets (including nested)
                            pattern = r'\([^()]*\)|\[[^\[\]]*\]'
                            while re.search(pattern, value):
                                value = re.sub(pattern, '', value)
                            value = re.sub(r'[:,.]', '', value)  # Remove colons and commas
                            value = re.sub(r'\s+', ' ', value).strip()  # Normalize whitespace
                            return value if value.lower() != 'nan' and value != '' else None
                        
                        # Drop unnecessary columns
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename and format columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)
                        child['PG SUBBRAND [-4427]'] = child['PG SUBBRAND [-4427]'].astype(str)
                        child['#US LOC BRAND [71177]'] = child['#US LOC BRAND [71177]'].astype(str)

                        # Replace AO subbrands with cleaned #US LOC BRAND
                        ao_mask = child['PG SUBBRAND [-4427]'].str.contains(r'\bAO\b', case=False, na=False)
                        child.loc[ao_mask, 'PG SUBBRAND [-4427]'] = (
                            child.loc[ao_mask, '#US LOC BRAND [71177]']
                            .apply(clean_brand_name)
                        )

                        # Replace "PRIVATE LABEL AO SB" with "PRIVATE LABEL"
                        child.loc[child['PG SUBBRAND [-4427]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4427]'] = 'PRIVATE LABEL'

                        # Replace "NOT APPLICABLE" with cleaned #US LOC BRAND
                        not_applicable_mask = child['PG SUBBRAND [-4427]'] == 'NOT APPLICABLE'
                        child.loc[not_applicable_mask, 'PG SUBBRAND [-4427]'] = (
                            child.loc[not_applicable_mask, '#US LOC BRAND [71177]']
                            .apply(clean_brand_name)
                        )

                        # Apply clean_brand_name to all remaining subbrand values
                        child['PG SUBBRAND [-4427]'] = child['PG SUBBRAND [-4427]'].apply(clean_brand_name)

                        # Handle PG SHAPE cleanup
                        child['#US LOC PAPER PLY [71144]'] = child['#US LOC PAPER PLY [71144]'].astype(str)
                        child.loc[child['#US LOC PAPER PLY [71144]'].str.match(r'(?i)^(NOT STATED|NOT COLLECTED|SAME UPC ALL PLYS)', na=False), '#US LOC PAPER PLY [71144]'] = np.nan

                        # MULTI PACK formatting
                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        # BASE SIZE formatting
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                        # Combine for aggregation
                        columns_to_combine = [
                            'PG SUBBRAND [-4427]', 'PG SHEET SIZE [-4426]', '#US LOC PAPER PLY [71144]',
                            'PG PACKAGE TYPE [-4424]', '#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]'
                        ]

                        # Apply clean_brand_name to all columns involved in aggregation
                        for col in columns_to_combine:
                            child[col] = child[col].apply(clean_brand_name)

                        # Build AGGREGATED_DATA excluding blanks and nan
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(
                            lambda row: ' '.join([str(x) for x in row if pd.notna(x) and str(x).strip().lower() != 'nan' and str(x).strip() != '']),
                            axis=1
                        )

                        # Drop unnamed columns
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG HAIR CARE":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Step 3: Replace NaNs with empty strings and remove '~'
                        child = child.replace(np.nan, '', regex=True)
                        child = child.applymap(lambda x: x.replace('~', '') if isinstance(x, str) else x)

                        # Step 4: Split BP and Non-BP
                        bp_upcs = child[child['BARCODE'].str.contains('BP', na=False, case=False)].copy()
                        non_bp_upcs = child[~child['BARCODE'].str.contains('BP', na=False, case=False)].copy()

                        # BRAND CLEANING FUNCTION
                        import re
                        def clean_brand_name(value):
                            if pd.isna(value):
                                return ''
                            value = str(value)
                            while re.search(r'\([^()]*\)', value):
                                value = re.sub(r'\([^()]*\)', '', value)
                            while re.search(r'\[[^\[\]]*\]', value):
                                value = re.sub(r'\[[^\[\]]*\]', '', value)
                            value = re.sub(r'[:,]', '', value)
                            value = value.replace('~', '')
                            return re.sub(r'\s+', ' ', value).strip()

                        # NON-BP RULES
                        def transform_non_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'NON_BP'

                            def handle_subbrand(row):
                                val = str(row['PG SUBBRAND [-6090]']).upper()
                                if 'PRIVATE LABEL AO SB' in val:
                                    return 'PRIVATE LABEL'
                                elif 'NOT APPLICABLE' in val:
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                elif any(x in val for x in ['AO SB', 'AO SUBBRAND']):
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG SUBBRAND [-6090]'])

                            df['PG SUBBRAND [-6090]'] = df.apply(handle_subbrand, axis=1)

                            # PG SCENT cleanup
                            df['PG BENEFIT [-6082]'] = df['PG BENEFIT [-6082]'].replace(['AO BENEFITS'], '')
                            df['PG STRENGTH [-6265]'] = df['PG STRENGTH [-6265]'].replace(['NOT APPLICABLE'], '')

                            # BASE SIZE replacements
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].astype(str)
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                            # MULTI PACK
                            df.loc[df['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = ''
                            df.loc[df['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                            # Aggregated value
                            aggregation_columns = [
                                'PG SUBBRAND [-6090]', 'PG BENEFIT [-6082]', 'PG STRENGTH [-6265]',
                                'PG SEGMENT [-6088]', '#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]'
                            ]

                            def aggregate_non_bp(row):
                                return ' '.join([
                                    str(row[col]).strip().replace(',', '') for col in aggregation_columns
                                    if str(row[col]).strip()
                                ])

                            df['AGGREGATED_VALUE'] = df.apply(aggregate_non_bp, axis=1)
                            return df

                        # BP RULES
                        def transform_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'BP'

                            def handle_brand(row):
                                val = str(row['PG BRAND [-6244]']).upper()
                                if 'PRIVATE LABEL AO' in val:
                                    return 'PRIVATE LABEL'
                                elif 'NOT APPLICABLE' in val:
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                elif 'AO' in val:
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG BRAND [-6244]'])

                            df['PG BRAND [-6244]'] = df.apply(handle_brand, axis=1)

                            # Remove BP suffix from BARCODE
                            df['BARCODE'] = df['BARCODE'].str.extract(r'(\d{12})')

                            # Aggregated value
                            def aggregate_bp(row):
                                parts = [row['PG BRAND [-6244]'], row['PG CATEGORY GIFT SET [-6250]']]
                                if row['PG BRAND [-6244]'] != 'PRIVATE LABEL':
                                    parts.append(row['BARCODE'])
                                return ' '.join([str(p).strip().replace(',', '') for p in parts if str(p).strip()])

                            df['AGGREGATED_VALUE'] = df.apply(aggregate_bp, axis=1)
                            return df

                        # Step 5: Apply transformations
                        transformed_non_bp = transform_non_bp(non_bp_upcs)
                        transformed_bp = transform_bp(bp_upcs)

                        final_df = final_df.loc[:, ~final_df.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG TOOTH BRUSH":
                                            
                        import re

                        # ============ FUNCTION TO CLEAN BRAND NAME ============ #
                        def clean_brand_name(value):
                            if pd.isna(value) or str(value).strip().lower() in ['nan', '']:
                                return None
                            value = str(value)
                            # Remove nested parentheses and brackets
                            pattern = r'\([^()]*\)|\[[^\[\]]*\]'
                            while re.search(pattern, value):
                                value = re.sub(pattern, '', value)
                            value = re.sub(r'[:,]', '', value)  # Remove colons, commas, periods
                            value = re.sub(r'\s+', ' ', value).strip()  # Normalize whitespace
                            return value if value.lower() != 'nan' and value != '' else None

                        # ============ BASIC CLEANUP ============ #
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # ============ REMOVE TILDES AND CLEAN WHITESPACE ============ #
                        child = child.applymap(lambda x: str(x).replace('~', '').strip() if isinstance(x, str) else x)

                        # ============ PG SUBBRAND REPLACEMENT LOGIC ============ #
                        mask1 = child['PG SUBBRAND [-5276]'].str.endswith('AO SB', na=False) & \
                                (child['PG SUBBRAND [-5276]'].str.upper() != 'PRIVATE LABEL AO SB')
                        mask2 = child['PG SUBBRAND [-5276]'].str.contains(r'\bAO SUBBRANDS\b', case=False, na=False)
                        mask = mask1 | mask2
                        child.loc[mask, 'PG SUBBRAND [-5276]'] = child.loc[mask, '#US LOC BRAND [71177]'].apply(clean_brand_name)
                        child.loc[child['PG SUBBRAND [-5276]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-5276]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-5276]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-5276]'] = child['#US LOC BRAND [71177]'].apply(clean_brand_name)

                        # ============ OTHER COLUMN TRANSFORMATIONS ============ #
                        # PG BRISTLE FIRMNESS
                        child['PG BRISTLE FIRMNESS [-5267]'] = child['PG BRISTLE FIRMNESS [-5267]'].astype(str)
                        child.loc[child['PG BRISTLE FIRMNESS [-5267]'].str.match(r'(?i)^NOT IDENTIFIED|NOT APPLICABLE', na=False), 'PG BRISTLE FIRMNESS [-5267]'] = np.nan

                        # PG MODEL NUMBER
                        child['PG MODEL NUMBER [-5293]'] = child['PG MODEL NUMBER [-5293]'].astype(str)
                        child.loc[child['PG MODEL NUMBER [-5293]'].str.match(r'(?i)^(AO|NOT APPLICABLE)', na=False), 'PG MODEL NUMBER [-5293]'] = np.nan

                        # LICENSED TRADEMARK: clean and fill from BP CONCAT LICENSED TRADEMARK if blank
                        child['#US LOC LICENSED TRADEMARK [71007]'] = child['#US LOC LICENSED TRADEMARK [71007]'].astype(str)
                        child.loc[
                            child['#US LOC LICENSED TRADEMARK [71007]'].str.strip().str.upper().isin(['', 'NOT STATED', 'NOT APPLICABLE', 'NOT COLLECTED', 'NAN']),
                            '#US LOC LICENSED TRADEMARK [71007]'
                        ] = np.nan

                        # Replace blank or NaN with values from BP CONCAT LICENSED TRADEMARK
                        child['#US LOC LICENSED TRADEMARK [71007]'] = child['#US LOC LICENSED TRADEMARK [71007]'].combine_first(child['#US LOC BP CONCAT LICENSED TRADEMARK [97894]'])

                        # MULTI PACK
                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        # BASE SIZE
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                        # ============ AGGREGATE COLUMNS ============ #
                        columns_to_combine = [
                            'PG SUBBRAND [-5276]', 'PG SEGMENT [-5294]', 'PG AUDIENCE [-5213]', 
                            'PG BRISTLE FIRMNESS [-5267]', 'PG MODEL NUMBER [-5293]', 
                            '#US LOC LICENSED TRADEMARK [71007]', '#US LOC BASE SIZE [71802]', 
                            '#US LOC IM MULTI CHAR [75626]'
                        ]

                        # Drop NaNs and blanks before joining
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(
                            lambda row: ' '.join(filter(None, [str(x).strip() for x in row if pd.notna(x) and str(x).strip().lower() != 'nan'])),
                            axis=1
                        )

                        # ============ FINAL CLEANUP ============ #
                        child.replace({np.nan: '', 'nan': ''}, inplace=True)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG SURFACE CARE":

                        import re

                        # ================= FUNCTION TO CLEAN BRAND NAME ================= #
                        def clean_brand_name(value):
                            if pd.isna(value) or str(value).strip().lower() in ['nan', '']:
                                return None
                            value = str(value)
                            pattern = r'\([^()]*\)|\[[^\[\]]*\]'
                            while re.search(pattern, value):
                                value = re.sub(pattern, '', value)
                            value = re.sub(r'[:,]', '', value)
                            value = re.sub(r'\s+', ' ', value).strip()
                            return value if value.lower() != 'nan' and value != '' else None
                        with zipfile.ZipFile(SKU_FILE.get(), 'r') as zipf:
                            filename = zipf.namelist()[0]
                            with zipf.open(filename) as file:
                                df = pd.read_csv(file, skiprows=1, dtype=str, low_memory=False)
                        # ================= BASIC CLEANUP ================= #
                        df.drop(df.columns[[1] + list(range(3, 8))], axis=1, inplace=True)
                        df.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)
                        df['BARCODE'] = df['BARCODE'].astype(str).str.zfill(12)
                        df = df.applymap(lambda x: str(x).replace('~', '').strip() if isinstance(x, str) else x)
                        df['#US LOC BRAND [71177]'] = df['#US LOC BRAND [71177]'].apply(clean_brand_name)
                        subbrand = 'PG SUBBRAND [-4839]'
                        df[subbrand] = df[subbrand].astype(str)
                        pl_ao_sb_mask = df[subbrand].str.contains(r'PRIVATE LABEL.*AO SB', case=False, na=False)
                        df.loc[pl_ao_sb_mask, subbrand] = 'PRIVATE LABEL'

                        # Replace AO-containing values (excluding PRIVATE LABEL AO SB) → cleaned #US LOC BRAND
                        ao_mask = df[subbrand].str.contains(r'\bAO\b', case=False, na=False)
                        df.loc[ao_mask & ~pl_ao_sb_mask, subbrand] = df.loc[ao_mask & ~pl_ao_sb_mask, '#US LOC BRAND [71177]']

# ================= BENEFIT, REFILL, PROMO, SCENT, SEGMENT, FORM CLEANING ================= #
                        df['PG BENEFIT [-4832]'] = df['PG BENEFIT [-4832]'].astype(str)
                        df.loc[df['PG BENEFIT [-4832]'].str.match(r'(?i)^N/A|AO BENEFITS', na=False), 'PG BENEFIT [-4832]'] = np.nan

                        df['PG REFILL [-4836]'] = df['PG REFILL [-4836]'].astype(str)
                        df.loc[df['PG REFILL [-4836]'].str.match(r'(?i)^N/A|NOT APPLICABLE', na=False), 'PG REFILL [-4836]'] = np.nan

                        df['PG PROMO ROTATION [-4852]'] = df['PG PROMO ROTATION [-4852]'].astype(str)
                        df.loc[df['PG PROMO ROTATION [-4852]'].str.match(r'(?i)^N/A|NOT APPLICABLE', na=False), 'PG PROMO ROTATION [-4852]'] = np.nan

                        df['PG SCENT [-4837]'] = df['PG SCENT [-4837]'].astype(str)
                        df.loc[df['PG SCENT [-4837]'].str.match(r'(?i)^N/A', na=False), 'PG SCENT [-4837]'] = np.nan

                        df['PG SEGMENT [-4838]'] = df['PG SEGMENT [-4838]'].astype(str)
                        df.loc[df['PG SEGMENT [-4838]'].str.match(r'(?i)^N/A|NOT APPLICABLE', na=False), 'PG SEGMENT [-4838]'] = np.nan

                        df['PG FORM [-4833]'] = df['PG FORM [-4833]'].astype(str)
                        df.loc[df['PG FORM [-4833]'].str.match(r'(?i)^AO FORMS', na=False), 'PG FORM [-4833]'] = np.nan
                        base_size_col = '#US LOC BASE SIZE [71802]'
                        df[base_size_col] = df[base_size_col].astype(str)
                        df[base_size_col] = df[base_size_col].str.replace(r'(?i)COUNT', 'CT', regex=True)
                        df[base_size_col] = df[base_size_col].str.replace(r'(?i)FLUID OUNCE|OUNCE', 'OZ', regex=True)

                        # ================= MULTIPACK ================= #
                        multi_col = '#US LOC IM MULTI CHAR [75626]'
                        df[multi_col] = df[multi_col].astype(str)
                        df.loc[df[multi_col] == '1', multi_col] = np.nan
                        df.loc[df[multi_col].str.match(r'^\d+$', na=False), multi_col] = df[multi_col] + ' PK'

                        # ================= QUICK CLEAN / NON QUICK CLEAN SPLIT ================= #
                        quick_mask = df['PG SUPER CATEGORY [-4840]'].str.upper() == 'QUICK CLEAN SYSTEMS'
                        quick = df[quick_mask].copy()
                        non_quick = df[~quick_mask].copy()

                        # ========== COLUMNS FOR AGGREGATION ========== #
                        quick_cols = [subbrand, 'PG BENEFIT [-4832]', 'PG SCENT [-4837]','PG REFILL [-4836]', 'PG PROMO ROTATION [-4852]', base_size_col, multi_col]
                        non_quick_cols = [subbrand, 'PG SCENT [-4837]', 'PG SEGMENT [-4838]', 'PG FORM [-4833]', base_size_col, multi_col]
                        # ========== AGGREGATED DATA ========== #
                        quick['AGGREGATED_DATA'] = quick[quick_cols].apply(
                            lambda row: ' '.join(filter(None, [str(x).strip() for x in row if pd.notna(x) and str(x).strip().lower() != 'nan'])),
                            axis=1
                        )

                        non_quick['AGGREGATED_DATA'] = non_quick[non_quick_cols].apply(
                            lambda row: ' '.join(filter(None, [str(x).strip() for x in row if pd.notna(x) and str(x).strip().lower() != 'nan'])),
                            axis=1
                        )

                        # ========== FINAL COMBINED OUTPUT ========== #
                        final_df = pd.concat([quick, non_quick], ignore_index=True)
                        final_df.replace({np.nan: '', 'nan': ''}, inplace=True)
                        final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')]



                   elif CATEGORY_button.get() == "PG PRE POST HAIR REMOVAL":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Step 3: Split BP and Non-BP
                        bp_upcs = child[child['BARCODE'].str.contains('BP', na=False, case=False)].copy()
                        non_bp_upcs = child[~child['BARCODE'].str.contains('BP', na=False, case=False)].copy()

                        # =============== BRAND CLEANING FUNCTION ===============
                        import re
                        def clean_brand_name(value):
                            if pd.isna(value):
                                return value
                            value = str(value)
                            # Remove nested parentheses and brackets
                            while re.search(r'\([^()]*\)', value):
                                value = re.sub(r'\([^()]*\)', '', value)
                            while re.search(r'\[[^\[\]]*\]', value):
                                value = re.sub(r'\[[^\[\]]*\]', '', value)
                            # Remove anything before colon
                            value = re.sub(r'^\s*[^:]+:\s*', '', value)
                            return re.sub(r'\s+', ' ', value).strip()

                        # =============== NON-BP RULES ===============
                        def transform_non_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'NON_BP'

                            def handle_subbrand(row):
                                val = str(row['PG SUBBRAND [-11065]']).upper()
                                if 'PRIVATE LABEL AO SB' in val:
                                    return 'PRIVATE LABEL'
                                elif 'NOT APPLICABLE' in val:
                                    return row['#US LOC BRAND [71177]']
                                elif any(x in val for x in ['AO SB', 'AO SUBBRAND', 'AO']):
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG SUBBRAND [-11065]'])

                            df['PG SUBBRAND [-11065]'] = df.apply(handle_subbrand, axis=1)

                            # PG HEAD TYPE cleanup
                            df['#US LOC SCENT [71469]'] = df['#US LOC SCENT [71469]'].replace(['NOT STATED','NOT COLLECTED'], np.nan)
                            df['PG FORM [-11177]'] = df['PG FORM [-11177]'].replace('AO FORMS', np.nan)

                            # BASE SIZE replacements (include FLUID OUNCE and FLUID OZ to OZ)
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].replace({
                                'COUNT': 'CT',
                                'OUNCE': 'OZ',
                                'FLUID OUNCE': 'OZ',
                                'FLUID OZ': 'OZ'
                            }, regex=True)

                            # MULTI PACK
                            df.loc[df['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = ''
                            df.loc[df['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                            # Aggregated value
                            aggregation_columns = [
                                'PG SUBBRAND [-11065]', '#US LOC SCENT [71469]', 'PG FORM [-11177]',
                                'PG SEGMENT [-10975]', '#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]'
                            ]

                            def aggregate_non_bp(row):
                                return ' '.join([
                                    str(row[col]).strip() for col in aggregation_columns
                                    if pd.notna(row[col]) and str(row[col]).strip()
                                ])

                            df['AGGREGATED_VALUE'] = df.apply(aggregate_non_bp, axis=1)

                            # Replace FLUID OUNCE and FLUID OZ with OZ in AGGREGATED_VALUE
                            df['AGGREGATED_VALUE'] = df['AGGREGATED_VALUE'].str.replace(r'\bFLUID\s+OUNCE(S)?\b', 'OZ', flags=re.IGNORECASE, regex=True)
                            df['AGGREGATED_VALUE'] = df['AGGREGATED_VALUE'].str.replace(r'\bFLUID\s+OZ\b', 'OZ', flags=re.IGNORECASE, regex=True)

                            return df

                        # =============== BP RULES ===============
                        def transform_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'BP'

                            def handle_brand(row):
                                val = str(row['PG BRAND [-11130]']).upper()
                                if 'PRIVATE LABEL AO' in val:
                                    return 'PRIVATE LABEL'
                                elif 'NOT APPLICABLE' in val:
                                    return row['#US LOC BRAND [71177]']
                                elif 'AO' in val:
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG BRAND [-11130]'])

                            df['PG BRAND [-11130]'] = df.apply(handle_brand, axis=1)

                            # Remove BP suffix from BARCODE
                            df['BARCODE'] = df['BARCODE'].str.extract(r'(\d{12})')

                            # Aggregated value
                            def aggregate_bp(row):
                                parts = [row['PG BRAND [-11130]'], row['PG CATEGORY GIFT SET [-11158]']]
                                if row['PG BRAND [-11130]'] != 'PRIVATE LABEL':
                                    parts.append(row['BARCODE'])
                                return ' '.join([str(p).strip() for p in parts if pd.notna(p) and str(p).strip()])

                            df['AGGREGATED_VALUE'] = df.apply(aggregate_bp, axis=1)

                            # Replace FLUID OUNCE and FLUID OZ with OZ in AGGREGATED_VALUE
                            df['AGGREGATED_VALUE'] = df['AGGREGATED_VALUE'].str.replace(r'\bFLUID\s+OUNCE(S)?\b', 'OZ', flags=re.IGNORECASE, regex=True)
                            df['AGGREGATED_VALUE'] = df['AGGREGATED_VALUE'].str.replace(r'\bFLUID\s+OZ\b', 'OZ', flags=re.IGNORECASE, regex=True)

                            return df

                        # Step 4: Apply transformations
                        transformed_non_bp = transform_non_bp(non_bp_upcs)
                        transformed_bp = transform_bp(bp_upcs)

                        # Step 5: Combine and export
                        final_df = pd.concat([transformed_non_bp, transformed_bp], ignore_index=True)

                   elif CATEGORY_button.get() == "PG DENTIFRICE AND WHITENING":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)


                        # Define function to clean #US LOC BRAND by removing nested brackets and content after colons
                        def remove_nested_parentheses(text):
                            if pd.isna(text):
                                return text

                            result = []
                            level = 0
                            for char in text:
                                if char == '(':
                                    level += 1
                                elif char == ')':
                                    if level > 0:
                                        level -= 1
                                elif level == 0:
                                    result.append(char)

                            return ''.join(result).strip()
                            return re.sub(r'\s+', ' ', ''.join(result)).strip()


                        mask1 = child['PG SUBBRAND [-4886]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-4886]'] != 'PRIVATE LABEL AO SB')

                        # Create a mask for rows where PG SUBBRAND [-4329] is exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-4886]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])

                        mask3 = child['PG SUBBRAND [-4886]'].str.startswith('AO', na=False)

                        mask4 = child['PG SUBBRAND [-4886]'].str.contains('AO', na = False)


                        # Combine both masks using OR (|)
                        mask = mask1 | mask2 | mask3 | mask4

                        # Replace with '#US LOC BRAND [71177]' while removing parentheses and content inside them
                        child.loc[mask, 'PG SUBBRAND [-4886]'] = child.loc[mask, '#US LOC BRAND [71177]'].apply(remove_nested_parentheses)


                        # If 'PG SUBBRAND [-4329]' is 'PRIVATE LABEL AO SB', replace it with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4886]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4886]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4886]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-4886]'] = child['#US LOC BRAND [71177]']

                        child['#US LOC FLAVOR [71057]'] = child['#US LOC FLAVOR [71057]'].astype(str)  # Convert to string type
                        child.loc[child['#US LOC FLAVOR [71057]'].str.match(r'(?i)^(NOT APPLICABLE|NOT COLLECTED|NOT STATED)', na=False), '#US LOC FLAVOR [71057]'] = np.nan

                        child['PG GROUP BENEFIT [-4892]'] = child['PG GROUP BENEFIT [-4892]'].astype(str)  # Convert to string type
                        child.loc[child['PG GROUP BENEFIT [-4892]'].str.match(r'(?i)^(AO GROUP BENEFITS)', na=False), 'PG GROUP BENEFIT [-4892]'] = np.nan

                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'


                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)  # Convert to string type

                        # Replace COUNT with CT
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)

                        # Replace FLUID OUNCE and OUNCE with OZ
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                        # Replace FLUID OUNCE and OUNCE with OZ
                        columns_to_combine = [
                            'PG SUBBRAND [-4886]', 'PG FORM [-4869]', '#US LOC FLAVOR [71057]', 
                            'PG GROUP BENEFIT [-4892]','#US LOC BASE SIZE [71802]', 
                            '#US LOC IM MULTI CHAR [75626]'
                        ]

                        child[columns_to_combine] = child[columns_to_combine].replace('nan', np.nan)

                        for col in columns_to_combine:
                            child[col] = child[col].str.replace(',', '', regex=False)

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(
                            lambda row: ' '.join(row.dropna().astype(str)), axis=1
                        )
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG AIRCARE":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        mask1 = child['PG SUBBRAND [-4613]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-4613]'] != 'PRIVATE LABEL AO SB')

                        # Create a mask for rows where PG SUBBRAND [-4329] is exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-4613]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])

                        # Combine both masks using OR (|)
                        mask = mask1 | mask2

                        # Replace with '#US LOC BRAND [71177]' while removing parentheses and content inside them
                        child.loc[mask, 'PG SUBBRAND [-4613]'] = (
                            child.loc[mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)  # Remove anything inside parentheses
                            .str.strip()
                        )

                        # If 'PG SUBBRAND [-4329]' is 'PRIVATE LABEL AO SB', replace it with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4613]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4613]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4613]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-4613]'] = child['#US LOC BRAND [71177]']

                        child['PG SCENT [-4560]'] = child['PG SCENT [-4560]'].astype(str)
                        child.loc[child['PG SCENT [-4560]'].str.match(r'(?i)^(NOT APPLICABLE)', na=False), 'PG SCENT [-4560]'] = np.nan 

                        child['PG FORM [-4558]'] = child['PG FORM [-4558]'].astype(str)
                        child.loc[child['PG FORM [-4558]'].str.match(r'(?i)^(AO FORMS)', na=False), 'PG FORM [-4558]'] = np.nan 

                        child['PG PACKAGE TYPE [-4624]'] = child['PG PACKAGE TYPE [-4624]'].astype(str)
                        child.loc[child['PG PACKAGE TYPE [-4624]'].str.match(r'(?i)^(1 COUNT|1 PACK)', na=False), 'PG PACKAGE TYPE [-4624]'] = np.nan 

                        # Replace FLUID OUNCE and OUNCE with OUNCES
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(
                            r'FLUID OUNCE|OUNCE', 'OUNCES', case=False, regex=True
                        )

                        # ===================== REMOVE TILDES (~) FROM ALL TEXT COLUMNS =====================
                        for col in child.select_dtypes(include='object').columns:
                            child[col] = child[col].str.replace('~', '', regex=False)

                        # Create AGGREGATED_DATA
                        columns_to_combine = [
                            'PG SUBBRAND [-4613]', 'PG SCENT [-4560]', 'PG PILLAR [-4629]', 
                            'PG FORM [-4558]', '#US LOC BASE SIZE [71802]', 'PG PACKAGE TYPE [-4624]'
                        ]

                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(
                            lambda row: ' '.join(row.dropna().astype(str)), axis=1
                        )

                        # Remove unnamed columns
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]
                   elif CATEGORY_button.get() == "PG PERSONAL CLEANSING":

                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        # Step 3: Split BP and Non-BP
                        bp_upcs = child[child['BARCODE'].str.contains('BP', na=False, case=False)].copy()
                        non_bp_upcs = child[~child['BARCODE'].str.contains('BP', na=False, case=False)].copy()

                        # =============== BRAND CLEANING FUNCTION ===============
                        import re
                        def clean_brand_name(value):
                            if pd.isna(value):
                                return value
                            value = str(value)
                            # Remove nested parentheses and brackets
                            while re.search(r'\([^()]*\)', value):
                                value = re.sub(r'\([^()]*\)', '', value)
                            while re.search(r'\[[^\[\]]*\]', value):
                                value = re.sub(r'\[[^\[\]]*\]', '', value)
                            # Remove all colons and commas
                            value = value.replace(':', '').replace(',', '')
                            return re.sub(r'\s+', ' ', value).strip()

                        # =============== NON-BP RULES ===============
                        def transform_non_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'NON_BP'

                            def handle_subbrand(row):
                                val = str(row['PG SUBBRAND [-4719]']).upper()
                                if 'PRIVATE LABEL AO SB' in val:
                                    return 'PRIVATE LABEL'
                                elif 'NOT APPLICABLE' in val:
                                    return row['#US LOC BRAND [71177]']
                                elif any(x in val for x in ['AO SB', 'AO SUBBRAND']):
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG SUBBRAND [-4719]'])

                            df['PG SUBBRAND [-4719]'] = df.apply(handle_subbrand, axis=1)

                            # PG SCENT cleanup
                            df['PG SCENT [-4543]'] = df['PG SCENT [-4543]'].replace(['NOT STATED', 'AO SCENTS', 'NOT COLLECTED'], np.nan)

                            # BASE SIZE replacements
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].astype(str)
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                            df['#US LOC BASE SIZE [71802]'] = df['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                            # MULTI PACK
                            df.loc[df['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = ''
                            df.loc[df['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                            # Aggregated value
                            aggregation_columns = [
                                'PG SUBBRAND [-4719]', 'PG SCENT [-4543]', 'PG SEGMENT [-4606]',
                                '#US LOC BASE SIZE [71802]', '#US LOC IM MULTI CHAR [75626]'
                            ]

                            def aggregate_non_bp(row):
                                return ' '.join([
                                    str(row[col]).strip().replace(',', '') for col in aggregation_columns
                                    if pd.notna(row[col]) and str(row[col]).strip()
                                ])

                            df['AGGREGATED_VALUE'] = df.apply(aggregate_non_bp, axis=1)
                            return df

                        # =============== BP RULES ===============
                        def transform_bp(df):
                            df = df.copy()
                            df['UPC_TYPE'] = 'BP'

                            def handle_brand(row):
                                val = str(row['PG BRAND [-4634]']).upper()
                                if 'PRIVATE LABEL AO' in val:
                                    return 'PRIVATE LABEL'
                                elif 'NOT APPLICABLE' in val:
                                    return row['#US LOC BRAND [71177]']
                                elif 'AO' in val:
                                    return clean_brand_name(row['#US LOC BRAND [71177]'])
                                else:
                                    return clean_brand_name(row['PG BRAND [-4634]'])

                            df['PG BRAND [-4634]'] = df.apply(handle_brand, axis=1)

                            # Remove BP suffix from BARCODE
                            df['BARCODE'] = df['BARCODE'].str.extract(r'(\d{12})')

                            # Aggregated value
                            def aggregate_bp(row):
                                parts = [row['PG BRAND [-4634]'], row['PG CATEGORY GIFT SET [-6118]']]
                                if row['PG BRAND [-4634]'] != 'PRIVATE LABEL':
                                    parts.append(row['BARCODE'])
                                return ' '.join([str(p).strip().replace(',', '') for p in parts if pd.notna(p) and str(p).strip()])

                            df['AGGREGATED_VALUE'] = df.apply(aggregate_bp, axis=1)
                            return df

                        # Step 4: Apply transformations
                        transformed_non_bp = transform_non_bp(non_bp_upcs)
                        transformed_bp = transform_bp(bp_upcs)

                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG SLEEP AIDS":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)
                        child.columns = child.columns.str.strip()

                        # Function to remove nested brackets from a string
                        import re
                        def remove_nested_brackets(text):
                            if pd.isna(text):
                                return text
                            while re.search(r'\([^()]*\)', text):
                                text = re.sub(r'\([^()]*\)', '', text)
                            while re.search(r'\[[^\[\]]*\]', text):
                                text = re.sub(r'\[[^\[\]]*\]', '', text)
                            return re.sub(r'\s{2,}', ' ', text).strip()

                        # ===== PG SUBBRAND [-4397] transformations =====
                        pg_subbrand_col = 'PG SUBBRAND [-4397]'
                        us_loc_brand_col = '#US LOC BRAND [71177]'

                        ao_mask = (
                            child[pg_subbrand_col].str.contains('AO SB', na=False) |
                            child[pg_subbrand_col].str.contains('AO SUBBRANDS', na=False)
                        )
                        child.loc[ao_mask, pg_subbrand_col] = (
                            child.loc[ao_mask, us_loc_brand_col].apply(remove_nested_brackets)
                        )

                        # ===== PG FORM [-4271] transformations =====
                        pg_form_col = 'PG FORM [-4271]'
                        us_loc_form_col = '#US LOC FORM [71782]'
                        ao_form_mask = child[pg_form_col].str.contains('AO FORM', na=False)
                        child.loc[ao_form_mask, pg_form_col] = (
                            child.loc[ao_form_mask, us_loc_form_col].apply(remove_nested_brackets)
                        )

                        # ===== FLAVOR [71057] transformations =====
                        child['#US LOC FLAVOR [71057]'] = child['#US LOC FLAVOR [71057]'].astype(str)
                        child.loc[child['#US LOC FLAVOR [71057]'].str.match(r'(?i)^(NOT APPLICABLE|NOT STATED|NOT COLLECTED)', na=False), '#US LOC FLAVOR [71057]'] = ''

                        # ===== BASE SIZE [71802] transformations =====
                        base_size_col = '#US LOC BASE SIZE [71802]'

                        def clean_base_size(val):
                            if pd.isna(val):
                                return val
                            val = val.upper()
                            if 'COUNT' in val:
                                return val.replace('COUNT', 'CT')
                            elif 'OUNCE' in val or 'FLUID OUNCE' in val:
                                return val.replace('FLUID OUNCE', 'OZ').replace('OUNCE', 'OZ')
                            return val

                        child[base_size_col] = child[base_size_col].apply(clean_base_size)

                        # ===== MULTI PACK [75626] transformations =====
                        multi_pack_col = '#US LOC IM MULTI CHAR [75626]'

                        def clean_multi_pack(val):
                            if pd.isna(val) or str(val).strip().lower() == 'nan':
                                return np.nan
                            
                            val = str(val).strip().upper()
                            
                            if val.isdigit():
                                num = int(val)
                                if num > 1:
                                    return f"{num} PK"
                                else:
                                    return np.nan
                            else:
                                match = re.match(r'^([2-9])\b', val)
                                if match and not val.endswith('PK'):
                                    return f"{match.group(1)} PK"
                                return val

                        child[multi_pack_col] = child[multi_pack_col].apply(clean_multi_pack)

                        # ===== AGGREGATED_DATA construction =====

                        # Columns to combine
                        columns_to_combine = [
                            'PG SUBBRAND [-4397]',
                            'PG SEGMENT [-4399]',
                            'PG FORM [-4271]',
                            'PG INGREDIENT [-4275]',
                            '#US LOC FLAVOR [71057]',
                            '#US LOC BASE SIZE [71802]',
                            '#US LOC IM MULTI CHAR [75626]'
                        ]

                        # Clean each value before joining
                        def clean_value(val):
                            if pd.isna(val):
                                return ''
                            return str(val).replace(':', '').strip()

                        def clean_aggregated_row(row):
                            cleaned = [clean_value(x) for x in row if pd.notna(x) and str(x).strip().lower() != 'nan']
                            return re.sub(r'\s{2,}', ' ', ' '.join(cleaned)).strip()

                        # Apply cleaning and aggregation
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(clean_aggregated_row, axis=1)

                        # Remove unwanted columns
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG RESPIRATORY":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns:
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        mask1 = child['PG SUBBRAND [-5085]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-5085]'] != 'PRIVATE LABEL AO SB')

                        # Create a mask for rows where PG SUBBRAND [-4329] is exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-5085]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])
                        child.loc[child['PG SUBBRAND [-5085]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-5085]'] = 'PRIVATE LABEL'


                        # Combine both masks using OR (|)
                        mask = mask1 | mask2

                        # Replace with '#US LOC BRAND [71177]' while removing parentheses and content inside them
                        child.loc[mask, 'PG SUBBRAND [-5085]'] = (
                            child.loc[mask, '#US LOC BRAND [71177]']
                            .str.replace(r"\(.*?\)", "", regex=True)  # Remove anything inside parentheses
                            .str.strip()
                        )

                        child['PG BENEFIT [-5091]'] = child['PG BENEFIT [-5091]'].str.strip().str.upper()

                        # Replace values containing "AO BENEFIT" or "NOT APPLICABLE" with NaN
                        child.loc[child['PG BENEFIT [-5091]'].str.contains(r'AO BENEFIT|NOT APPLICABLE', na=False, case=False, regex=True), 'PG BENEFIT [-5091]'] = np.nan

                        child['PG SUBFLAVOR [-40578]'] = child['PG SUBFLAVOR [-40578]'].astype(str).str.strip().str.upper()

                        # Replace values containing "AO FLAVOR", "FLAVOR UNDEFINED", or "UNFLAVORED" with NaN (empty)
                        child.loc[child['PG SUBFLAVOR [-40578]'].str.contains(r'AO FLAVOR|FLAVOR UNDEFINED|UNFLAVORED', na=False, case=False, regex=True), 'PG SUBFLAVOR [-40578]'] = np.nan

                        child['PG SUBFORM [-40595]'] = child['PG SUBFORM [-40595]'].astype(str).str.strip().str.upper()

                        # Replace values containing "AO FORMS" with NaN (empty)
                        child.loc[child['PG SUBFORM [-40595]'].str.contains(r'AO FORMS', na=False, case=False, regex=True), 'PG SUBFORM [-40595]'] = np.nan

                        child['PG INGREDIENT [-5080]'] = child['PG INGREDIENT [-5080]'].astype(str).str.strip().str.upper()

                        # Replace values containing "NO DECONGESTANT" or "AO INGREDIENTS" with NaN (empty)
                        child.loc[child['PG INGREDIENT [-5080]'].str.contains(r'NO DECONGESTANT|AO INGREDIENTS|COMBO-PACK|INGREDIENTS UNDEFINED', na=False, case=False, regex=True), 'PG INGREDIENT [-5080]'] = np.nan
                        # Ensure 'BASE SIZE' column is a string
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)

                        # Replace 'COUNT' with 'CT' and 'OUNCE' with 'OZ'
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)

                        # Replace 'COUNT' with 'CT' and 'OUNCE' with 'OZ'
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'OUNCE', 'OZ', case=False, regex=True)
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str).str.replace(r'FLUID OZ', 'OZ', case=False, regex=True)



                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        child.loc[child['PG TIME [-5086]'].str.contains(r'COMBO-PACK', na=False, case=False, regex=True), 'PG TIME [-5086]'] = np.nan


                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        columns_to_combine = [
                            'PG SUBBRAND [-5085]', 'PG SEGMENT [-5084]', 
                            'PG BENEFIT [-5091]', 'PG SUBFLAVOR [-40578]','PG SUBFORM [-40595]',
                            'PG TIME [-5086]','PG INGREDIENT [-5080]','#US LOC BASE SIZE [71802]','#US LOC IM MULTI CHAR [75626]'
                            ]

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]
                        child['AGGREGATED_DATA'] = child['AGGREGATED_DATA'].str.replace(r'\bnan\b', '', regex=True).str.strip()

                   elif CATEGORY_button.get() == "PG FEMININE CARE":
                        import re

                        def reformat_brand_name(name):
                            if pd.isna(name):
                                return name
                            # Extract all bracketed content (handles multiple brackets)
                            parts = re.findall(r'\((.*?)\)', name)
                            outside = re.sub(r'\(.*?\)', '', name).strip()
                            outside = re.sub(r'[^\w\s]', '', outside)  # remove punctuation like periods
                            reformatted = ' '.join(parts + [outside]).strip()
                            return reformatted

                        # Drop unnecessary columns
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)

                        ao_mask = (
                            child['PG SUBBRAND [-4505]'].str.contains(r'\bAO\b', case=False, na=False) &
                            ~child['PG SUBBRAND [-4505]'].str.contains(r'PRIVATE LABEL AO SB', case=False, na=False)
                        )

                        # Function to remove all bracketed content, including nested ones
                        def remove_brackets(text):
                            while True:
                                new_text = re.sub(r'\[[^\[\]]*\]|\([^\(\)]*\)', '', text)
                                if new_text == text:
                                    break
                                text = new_text
                            return text.strip()

                        import re
                        # Clean #US LOC BRAND values by removing all bracketed content
                        child.loc[ao_mask, 'PG SUBBRAND [-4505]'] = child.loc[ao_mask, '#US LOC BRAND [71177]'].apply(lambda x: remove_brackets(str(x)))

                        # Replace 'PRIVATE LABEL AO SB' with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4505]'].str.upper() == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4505]'] = 'PRIVATE LABEL'

                        # Replace 'NOT APPLICABLE' in PG SUBBRAND with cleaned #US LOC BRAND
                        child.loc[child['PG SUBBRAND [-4505]'].str.upper() == 'NOT APPLICABLE', 'PG SUBBRAND [-4505]'] = (
                            child.loc[child['PG SUBBRAND [-4505]'].str.upper() == 'NOT APPLICABLE', '#US LOC BRAND [71177]']
                            .apply(lambda x: remove_brackets(str(x)))
                        )

                        #To exclude the value in PG FLAVOR, if it has AO SB in it
                        child['PG SCENT [-4502]'] = child['PG SCENT [-4502]'].astype(str)  # Convert to string type
                        child.loc[child['PG SCENT [-4502]'].str.match(r'(?i)^(NOT APPLICABLE)', na=False), 'PG SCENT [-4502]'] = np.nan  # Replace values

                        child['PG COVERAGE [-4498]'] = child['PG COVERAGE [-4498]'].astype(str)  # Convert to string type
                        child.loc[child['PG COVERAGE [-4498]'].str.match(r'(?i)^(NOT APPLICABLE)', na=False), 'PG COVERAGE [-4498]'] = np.nan  # Replace values

                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'

                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)  # Convert to string type

                        # Replace COUNT with CT
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)

                        # Replace FLUID OUNCE and OUNCE with OZ
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)
                        columns_to_combine = [
                            'PG SUBBRAND [-4505]', 'PG SEGMENT [-4503]', 'PG COVERAGE [-4498]', 
                            'PG SCENT [-4502]', '#US LOC BASE SIZE [71802]', 
                            '#US LOC IM MULTI CHAR [75626]'
                        ]

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]

                   elif CATEGORY_button.get() == "PG BLEACH":
                        child.drop(child.columns[[1] + list(range(3, 8))], axis=1, inplace=True)

                        # Rename columns
                        child.rename(columns={'0': 'NAN_KEY', 'Unnamed: 2': 'BARCODE'}, inplace=True)

                        # Format BARCODE to 12 digits by padding with zeros
                        child['BARCODE'] = child['BARCODE'].astype(str).str.zfill(12)


                        # Define function to clean #US LOC BRAND by removing nested brackets and content after colons
                        import re
                        def remove_nested_parentheses(text):
                            if pd.isna(text):
                                return text

                            result = []
                            level = 0
                            for char in text:
                                if char == '(':
                                    level += 1
                                elif char == ')':
                                    if level > 0:
                                        level -= 1
                                elif level == 0:
                                    result.append(char)

                            return ''.join(result).strip()
                            return re.sub(r'\s+', ' ', ''.join(result)).strip()


                        mask1 = child['PG SUBBRAND [-4141]'].str.endswith('AO SB', na=False) & (child['PG SUBBRAND [-4141]'] != 'PRIVATE LABEL AO SB')

                        # Create a mask for rows where PG SUBBRAND [-4329] is exactly 'AO SUBBRANDS' or 'AO SUBBRANDS CHILD'
                        mask2 = child['PG SUBBRAND [-4141]'].isin(['AO SUBBRANDS', 'AO SUBBRANDS CHILD'])

                        mask3 = child['PG SUBBRAND [-4141]'].str.startswith('AO', na=False)

                        mask4 = child['PG SUBBRAND [-4141]'].str.contains('AO', na = False)


                        # Combine both masks using OR (|)
                        mask = mask1 | mask2 | mask3 | mask4

                        # Replace with '#US LOC BRAND [71177]' while removing parentheses and content inside them
                        child.loc[mask, 'PG SUBBRAND [-4141]'] = child.loc[mask, '#US LOC BRAND [71177]'].apply(remove_nested_parentheses)


                        # If 'PG SUBBRAND [-4329]' is 'PRIVATE LABEL AO SB', replace it with 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4141]'] == 'PRIVATE LABEL AO SB', 'PG SUBBRAND [-4141]'] = 'PRIVATE LABEL'
                        child.loc[child['PG SUBBRAND [-4141]'] == 'NOT APPLICABLE', 'PG SUBBRAND [-4141]'] = child['#US LOC BRAND [71177]']

                        child['#US LOC SCENT [71469]'] = child['#US LOC SCENT [71469]'].astype(str)  # Convert to string type
                        child.loc[child['#US LOC SCENT [71469]'].str.match(r'(?i)^(NOT APPLICABLE|NOT COLLECTED|NOT STATED|N/A)', na=False), '#US LOC SCENT [71469]'] = np.nan

                        child['PG PACKAGE TYPE [-4137]'] = child['PG PACKAGE TYPE [-4137]'].astype(str)  # Convert to string type
                        child.loc[child['PG PACKAGE TYPE [-4137]'].str.match(r'(?i)^(AO PACKAGE TYPES)', na=False), 'PG PACKAGE TYPE [-4137]'] = np.nan

                        child['#US LOC IM MULTI CHAR [75626]'] = child['#US LOC IM MULTI CHAR [75626]'].astype(str)  # Ensure column is treated as a string

                        # Remove "1 MULTI"
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'] == '1', '#US LOC IM MULTI CHAR [75626]'] = np.nan  

                        # Replace "MULTI" with "PK" for values greater than 1 (e.g., "2 MULTI" -> "2PK")
                        child.loc[child['#US LOC IM MULTI CHAR [75626]'].str.match(r'^\d+$', na=False) & (child['#US LOC IM MULTI CHAR [75626]'] != '1'), '#US LOC IM MULTI CHAR [75626]'] += ' PK'


                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].astype(str)  # Convert to string type

                        # Replace COUNT with CT
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'COUNT', 'CT', case=False, regex=True)

                        # Replace FLUID OUNCE and OUNCE with OZ
                        child['#US LOC BASE SIZE [71802]'] = child['#US LOC BASE SIZE [71802]'].str.replace(r'FLUID OUNCE|OUNCE', 'OZ', case=False, regex=True)

                        # Replace FLUID OUNCE and OUNCE with OZ
                        columns_to_combine = [
                            'PG SUBBRAND [-4141]', '#US LOC SCENT [71469]', 'PG SEGMENT [-4139]', 
                            'PG PACKAGE TYPE [-4137]','#US LOC BASE SIZE [71802]', 
                            '#US LOC IM MULTI CHAR [75626]'
                        ]

                        child[columns_to_combine] = child[columns_to_combine].replace('nan', np.nan)

                        for col in columns_to_combine:
                            child[col] = child[col].str.replace(',', '', regex=False)

                        # Create a new column for aggregated data
                        child['AGGREGATED_DATA'] = child[columns_to_combine].apply(
                            lambda row: ' '.join(row.dropna().astype(str)), axis=1
                        )
                        child = child.loc[:, ~child.columns.str.startswith('Unnamed')]




    try:
        if CATEGORY_button.get() == "PG SURFACE CARE":
            final_df = pd.concat([quick, non_quick], ignore_index=True)
            final_df.replace({np.nan: '', 'nan': ''}, inplace=True)
            final_df = final_df.loc[:, ~final_df.columns.str.contains('^Unnamed')]
            output_filepath = os.path.join(SKU_FILE_PATH.get(),f"MODIFIED {CATEGORY_button.get()}.xlsx")
            final_df.to_excel(output_filepath, index=False)
        else:
            final_df = pd.concat([transformed_non_bp, transformed_bp], ignore_index=True)
            output_filepath = os.path.join(SKU_FILE_PATH.get(),f"MODIFIED {CATEGORY_button.get()}.xlsx")
            final_df.to_excel(output_filepath, index=False)
    except:
        output_filepath = os.path.join(SKU_FILE_PATH.get(),f"MODIFIED {CATEGORY_button.get()}.xlsx")
        child.to_excel(output_filepath, index=False)

# File Upload
SKU_Title = CTkLabel(app,text = "SKU FILE",font=('Consolas',15),text_color="white")
SKU_Title.place(relx=0.1,rely=0.1)

SKU_Entry = CTkEntry(app,textvariable=SKU_FILE_NAME,width=350,height=30,text_color="#FFC50D",font=("Consolas",15),corner_radius=4)
SKU_Entry.place(relx=0.27,rely=0.1)

SKU_button = CTkButton(app,text=" BROWSER ",width=10,height=10,command=SKU,corner_radius=45)
SKU_button.place(relx=0.84,rely=0.1)


Client_Name = CTkLabel(app,text = "CATEGORY NAME",font=('Consolas',15),text_color="white")
Client_Name.place(relx=0.1,rely=0.23)

CATEGORY_button =CTkComboBox(app,values=("PG DEODORANT","PG BLEACH","PG FEMININE CARE","PG RESPIRATORY","PG SLEEP AIDS","PG PERSONAL CLEANSING","PG AIRCARE","PG DENTIFRICE AND WHITENING","PG PRE POST HAIR REMOVAL","PG SURFACE CARE","PG TOOTH BRUSH","PG HAIR CARE","PG PAPER TOWELS","PG ORAL","PG LAUNDRY CARE","PG DISH CARE","PG DIGESTIVE HEALTH","PG DIAPERS","PG FACIAL SKINCARE","PG DENTURE CARE","PG BABY AND KID WIPES","PG BLADES AND RAZORS","PG SELF DIAGNOSTIC TISSUE","PG BATH TISSUE","PG FACIAL TISSUE","PG ADULT INCONTINENCE","PG APP STYLING","PG PEST CONTROL","PG LAWN CARE","PG FLOSS","PG FABRIC CONDITIONER","PG STOMACH REMEDIES"),width=180, height=20,corner_radius=4,text_color="#FFC50D")
CATEGORY_button.place(relx=0.27,rely=0.23)

SKU_Run = CTkButton(app,text=" RUN ",width=10,height=10,command=SKU_Smplifiy,corner_radius=45)
SKU_Run.place(relx=0.84,rely=0.23)


app.mainloop()
