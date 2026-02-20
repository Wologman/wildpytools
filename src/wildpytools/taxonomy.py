'''Useful classes and functions to map various bird-naming strategies between 
Department of Conservation, ebird, scientific naming, & ML model predictions

I'm assuming from the start that we're not going to work with a class that doesn't 
have a scientific name, and potentially that name might go to subspecies or hybrids.  
But that we may want to merge species or sub-species for a variety of reasons to a 
single eBird code.

These mappings should be consistent with the following sources:

https://www.nzor.org.nz/names/
https://nzbirdsonline.org.nz/ 
https://ebird.org/  (Mapping from scientific name to ebird code only)
'''

import pandas as pd
import random


class BirdNamer:
    '''Handles bird name conversions for anything that already has an ebird code
    There must be a one-one relationship between scientific name and ebird code
    The incoming mapping dict will use ebird code as keys, scientific and common 
    names as values
    
    Requires a csv file with the following header columns:
    CommonName	eBird	ScientificName	ExtraName

    '''
    
    def __init__(self, mapping_df):
        #modify this to ensure that the extra_name column of the mapping_df is filled
        if 'ExtraName' not in mapping_df.columns:
            mapping_df = mapping_df.copy()  # avoid modifying the original df
            mapping_df['ExtraName'] = mapping_df['CommonName']
        self.bird_list = mapping_df['eBird'].tolist()
        self.short_names_list = sorted(list(set(mapping_df['ExtraName'])))
        self.long_names_list = mapping_df['CommonName'].tolist()
        self.science_names_list = mapping_df['ScientificName'].tolist()
        self.mapping_dict = mapping_df.set_index('eBird').apply(tuple, axis=1).to_dict()
         
    def vec_to_ebird(self, binary_array):
        '''Should convert any array-like input into a list of birds present with ebird codes'''
        return [self.bird_list[i] for i, value in enumerate(binary_array) if value == 1]
    
    def common_name(self, key):
        '''Returns the common name from the e-bird code'''
        if self.mapping_dict is not None:
            value = self.mapping_dict.get(key, key)
            if isinstance(value, tuple):
                return value[0]
            else:
                return value
        else:
            return key
               
    def scientific_name(self, key):
        '''Returns the scientific name from the e-bird code'''
        if self.mapping_dict is not None:
            value = self.mapping_dict.get(key, key)
            if isinstance(value, tuple):
                return value[1]
            else:
                return value
        else:
            return key


    def scientific_name(self, key):
        '''Returns the extra name from the e-bird code'''
        if self.mapping_dict is not None:
            value = self.mapping_dict.get(key, key)
            if isinstance(value, tuple):
                return value[1]
            else:
                return value
        else:
            return key
  
    def extra_name(self, key):
        '''Returns an additional common name from the e-bird code
        Could be used to consoldiate less specific names.  
        For example Great Spotted Kiwi & Little Splotted Kiwi => Spotted Kiwi'''
        if self.mapping_dict is not None:
            value = self.mapping_dict.get(key, key)
            if isinstance(value, tuple):
                if value[2]:
                    return value[2] #return the extra name
                else:
                    return value[0] #return the common name
            else:
                return value
        else:
            return key

    def common_names(self, key_list):
        '''Returns a list of common names from a list of e-bird names'''
        common_list=[]
        for key in key_list:
            if self.mapping_dict is not None:
                value = self.mapping_dict.get(key, key)
                if isinstance(value, tuple):
                    common_list.append(value[0])
                else:
                    common_list.append(value)
            else:
                common_list.append(key)
        return common_list

    def scientific_names(self, key_list):
        '''Returns a list of scientific names from a list of e-bird names'''
        sci_list=[]
        for key in key_list:
            if self.mapping_dict is not None:
                value = self.mapping_dict.get(key, key)
                if isinstance(value, tuple):
                    sci_list.append(value[1])
            else:
                sci_list.append(key)
        return sci_list
    
    def extra_names(self, key_list):
        '''Returns a list of scientific names from a list of e-bird names'''
        sci_list=[]
        for key in key_list:
            if self.mapping_dict is not None:
                value = self.mapping_dict.get(key, key)
                if isinstance(value, tuple):
                    sci_list.append(value[2])
            else:
                sci_list.append(key)
        return sci_list



class BirdCodeConverter:
    '''Takes a path to csv file with the headers 'Code', 'CommonName', 'eBird', 'ScientificName',
      Returns a dictionary mapping the 'Code' to the other three.  The goal is to take some
      mystery internal organisation system (eg. from DOC) and make an attribute that converts 
      that code to an eBird code 
    '''
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        grouped_df = df.groupby('Code').first()
        ebird_gr_df = df.groupby('eBird').first()
        self.ebird = {code: row['eBird'] for code, row in grouped_df.iterrows()}
        self.scibird = {code: row['ScientificName'] for code, row in grouped_df.iterrows()}
        self.combird = {code: row['CommonName'] for code, row in grouped_df.iterrows()}
        self.com_sci = {ebird:(row['CommonName'],row['ScientificName']) for ebird, row in ebird_gr_df.iterrows()}


def get_name_map_dict(df):
    '''Maps the common names and scientific names to the ebird code from the labels 
    dataframe This is intended to work with Kaggle Datasets, which have, 
    'primary_label', 'common_name' and 'scientific_name' columns.
    '''
    all_names = {}
    grouped_df = df.groupby('primary_label').first()
    next_dict = dict(zip(grouped_df.index, (grouped_df['common_name'], grouped_df['scientific_name']))) 
    for key, value in next_dict.items():
        if key not in all_names:
            all_names[key] = value
    random_entries = random.sample(list(all_names.items()), 6)
    for entry in random_entries:
        print(entry)
    return all_names


def merge_names(map_dict, merger_dict):
    '''Takes a name mapping dict, in the form {ebird:(common_name, scientific_name)}
    and and merges the ebird classes {ebird_1:new_ebird, ebird_2:new_ebird}
    for example to merge Snares Crested Penguins and Yellow Eyed Penguins:
    {snapen1_yeepen1:(snapen_1, yeepen1)}, The common and scientific
    common values will become longer strings: 'Snares Crested Penguin or Yellow Eyed Penguin'
    scientific values will become: 'Eudyptes robustus or Megadyptes antipodes'
    Returns the new mapping dictionary, maintaining 1:1 between ebird and the name values'''
    
    old_keys = {item for value in merger_dict.values() for item in value}
    to_merge = {key: map_dict.pop(key) for key in old_keys if key in map_dict}

    new_map = {}
    for key,value in merger_dict.items():
        new_common_names = ' or '.join([to_merge[val][0] for val in value])
        new_sci_names = ' or '.join([to_merge[val][1] for val in value])
        new_map[key] = (new_common_names, new_sci_names)

    return map_dict.update(new_map)


def map_names(row, mapping):
    common_name = mapping[row['primary_label']][0]
    scientific_name = mapping[row['primary_label']][1]
    return pd.Series([common_name, scientific_name], index=['common_name', 'scientific_name'])

#df[['common_name', 'scientific_name']] = df.apply(map_names, axis=1)