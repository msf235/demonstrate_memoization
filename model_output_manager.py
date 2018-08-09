import os
import numpy as np
import pandas as pd
import h5py

# %% Hyperparameters
data_file_name = "model_data"

# %% Helper functions
def __unique_to_set(a, b):
    """
    Return elements that are unique to container a and items that are unique to container b among the union of a and b.

    Args:
        a (container):
        b (container):

    Returns:
        a_unique (list): elements that are contained in container a but not container b
        b_unique (list): elements that are contained in container b but not container a

    """

    def overlap(a, b):
        return list(set(a) & set(b))

    def difference(a, b):
        return list(set(a) ^ set(b))

    dif = difference(a, b)
    a_unique = overlap(a, dif)
    b_unique = overlap(b, dif)
    return a_unique, b_unique

# %% Methods for outputing data

def update_output_table(table_params, table_path='output/param_table.csv', compare_exclude=[], column_labels=None,
                        overwrite_existing=True):
    """
    Add row to output table from param_dict.

    Args:
        table_params (dict, OrderedDict): Parameters that will be put into the table
        compare_exclude (list): Parameters that will be excluded when determining if two rows represent the same
            run. For instance, if runs are identical except for the date when the run was done, then it might be
            reasonable to consider the runs as being identical, reflected in the variable run_number. Hence,
            one may want to put the date parameter key in compare_exclude.
        table_path (string): The filepath for the table (including that table name, i.e. 'output/param_table.csv')
        column_labels (list): Contains the keys of params_table in the order in which they should be written in the
            output table.
        overwrite_existing (bool): Whether or not to overwrite identical table entries or make a new row and
        increment run_id.

    Returns:

    """
    filepath = table_path.split('/')
    filepath = '/'.join(filepath[:-1])
    if not os.path.isdir(filepath):
        os.mkdir(filepath)
    if column_labels is None:
        column_labels = list(table_params.keys()).copy()
    if 'run_number' not in column_labels:
        column_labels.append('run_number')

    # run_id = uuid.uuid4().hex
    if not os.path.isfile(table_path):
        run_id = 0
        param_df = pd.DataFrame(table_params, index=[run_id], columns=column_labels)
        param_df['run_number'] = 0
        param_df.to_csv(table_path)
    else:
        param_df = pd.read_csv(table_path, index_col=0)
        new_cols = __unique_to_set(param_df.columns, column_labels)[1]  # param_keys that don't yet belong to param_df
        for key in new_cols:
            param_df[key] = pd.Series(None, index=param_df.index)
        unique_to_param_df = __unique_to_set(param_df.columns, column_labels)[0]
        if not unique_to_param_df:  # If column_labels is comprehensive
            param_df = param_df[column_labels]  # Reorder colums of param_df based on column_labels
        run_id = np.max(np.array(param_df.index)) + 1
        new_row = pd.DataFrame(table_params, index=[run_id])
        for e1 in unique_to_param_df:  # Add placeholders to new row for items that weren't in table_params
            new_row[e1] = np.nan
        # new_row = new_row[column_labels]
        compare_exclude2 = compare_exclude.copy()
        compare_exclude2.append('run_number')
        temp1 = param_df.drop(compare_exclude2, axis=1, errors='ignore')
        temp2 = new_row.drop(compare_exclude, axis=1, errors='ignore')
        temp_merge = pd.merge(temp1, temp2)
        # This is needed to ensure proper order in some cases (if table_params has less items than the table has
        #   columns)
        column_labels = list(temp_merge.columns)
        column_labels.append('run_number')
        run_number = temp_merge.shape[0]

        if run_number == 0 or not overwrite_existing:
            new_row['run_number'] = run_number
            new_row = new_row[column_labels]
            param_df = param_df.append(new_row)
            param_df.to_csv(table_path)
        else:
            temp_merge = temp1.reset_index().merge(temp2).set_index('index')
            run_id = np.max(np.array(temp_merge.index))
    return run_id

def dir_for_run(table_params, run_name, table_path='output/param_table.csv', compare_exclude=[],
                column_labels=None, overwrite_existing=True):
    run_id = update_output_table(table_params, table_path, compare_exclude, column_labels, overwrite_existing)

    table_dir = '/'.join(table_path.split('/')[:-1])
    run_dir = table_dir + '/' + run_name + '_' + str(run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_id, run_dir

def write_output(output, params, table_params, output_path, overwrite=False, data_filetype='hdf5'):
    """

    Args:
        output (dict): Dictionary that holds the output data
        params (dict): Dictionary that holds the parameters
        table_params (dict): Dictionary that holds the parameters in the output table
        output_path (string): Filepath for output file

    Returns:

    """
    output_dir = output_path.split('/')
    output_dir = '/'.join(output_dir[:-1])
    print()
    print("Attempting to write data to '" + os.getcwd() + '/' + output_path + "'")
    print()
    write_output = True
    try:
        os.makedirs(output_dir, exist_ok=False)
    except (OSError, FileExistsError):
        if overwrite:
            print("Warning: existing data directory overwritten.")
        else:
            print("Data directory already exists. Not writing output.")
            write_output = False
    if write_output:
        if data_filetype == 'hdf5':
            with h5py.File(output_path, "w") as fid:
                param_grp = fid.create_group("parameters")
                param_table_grp = fid.create_group("table_parameters")
                out_grp = fid.create_group("output")
                for key in params:
                    if params[key] is not None:
                        param_grp.create_dataset(key, data=params[key])
                for key in table_params:
                    if table_params[key] is not None:
                        param_table_grp.create_dataset(key, data=table_params[key])
                for key in output:
                    if output[key] is not None:
                        out_grp.create_dataset(key, data=output[key])
        elif data_filetype == 'pickle':
            import pickle as pkl
            data = dict(parameters=params, table_parameters=table_params, output=output)
            with open(output_path, "wb") as fid:
                pkl.dump(data, fid)

        print("Done. Data written.")

def save_model(table_params, table_path, model_output, params, run_name="", compare_exclude=[],
               columns=None, overwrite_existing=False, data_filetype='hdf5'):
    """

    Creates an entry in the output_table and saves the output in the corresponding directory. Basically just a
    wrapper to call update_output_table and then write_output.

    Args:
        table_params (dict, OrderedDict): Parameters that will be put into the table
        compare_exclude (list): Parameters that will be excluded when determining if two rows represent the same
            run. For instance, if runs are identical except for the date when the run was done, then it might be
            reasonable to consider the runs as being identical, reflected in the variable run_number. Hence,
            one may want to put the date parameter key in compare_exclude.
        table_path (string): The filepath for the table.
        model_output (dict): Dictionary that holds the output data
        params (dict): Dictionary that holds the parameters
        run_name (str): Name to give this
        columns (list): Contains the keys of params_table in the order in which they should be written in the
            output table.
        overwrite_existing (bool): Whether or not to overwrite identical table entries or make a new row and
            increment run_id.
        output_path (string): Filepath for output file
        data_filetype (str): Filetype for data to be written in. Currently only hdf5 is supported.

    Returns:

    """
    run_id = update_output_table(table_params, table_path, compare_exclude, columns, overwrite_existing)
    table_dir = table_path.split('/')
    table_dir = '/'.join(table_dir[:-1])
    if data_filetype == 'hdf5':
        file_name = data_file_name + '.h5'
    elif data_filetype == 'pickle':
        file_name = data_file_name + '.pkl'
    output_dir = table_dir + '/' + run_name + '_' + str(run_id)
    output_path = output_dir + '/' + file_name
    params.update(dict(table_path=table_path, output_dir=output_dir, run_id=run_id))
    write_output(model_output, params, table_params, output_path, overwrite_existing, data_filetype)

    return run_id, output_path

# %% Methods for loading data
# Todo: Build in support for nested dictionaries / groups
def hdf5group_to_dictionary(h5grp):
    d = {}
    for key in h5grp:
        d[key] = h5grp[key].value
    return d

def run_with_id_exists(run_name, run_id, table_dir='output'):
    """
    Given the name of the run, the ID of the run, and the directory of the output table, checks to see if the run
    exists.

    Args:
        run_name ():
        run_id ():
        table_dir ():

    Returns:

    """
    filename = table_dir + '/' + run_name + '_' + str(run_id) + '/data.hdf5'
    return os.path.exists(filename)

def run_with_params_exists(table_params, table_path='output/param_table.csv', compare_exclude=[]):
    """
    Given a set of parameters, check if a run matching this set exists.

    Args:
        table_params (dict, OrderedDict): Parameters that will be put into the table
        table_path (string): The filepath for the table.
        compare_exclude (list): Parameters that will be excluded when determining if two rows represent the same
            run. For instance, if runs are identical except for the date when the run was done, then it might be
            reasonable to consider the runs as being identical, reflected in the variable run_number. Hence,
            one may want to put the date parameter key in compare_exclude.

    Returns:

    """

    # run_id = uuid.uuid4().hex
    if not os.path.exists(table_path):
        return False
    column_labels = list(table_params.keys()).copy()
    if 'run_number' not in column_labels:
        column_labels.append('run_number')
    param_df = pd.read_csv(table_path, index_col=0)
    new_cols = __unique_to_set(param_df.columns, column_labels)[1]  # param_keys that don't yet belong to param_df
    for key in new_cols:
        param_df[key] = pd.Series(None, index=param_df.index)
    unique_to_param_df = __unique_to_set(param_df.columns, column_labels)[0]
    if not unique_to_param_df:  # If column_labels is comprehensive
        param_df = param_df[column_labels]  # Reorder colums of param_df based on column_labels

    run_id = np.max(np.array(param_df.index)) + 1
    new_row = pd.DataFrame(table_params, index=[run_id])
    for e1 in unique_to_param_df:  # Add placeholders to new row for items that weren't in param_dict
        new_row[e1] = np.nan
    # new_row = new_row[column_labels]

    compare_exclude2 = compare_exclude.copy()
    compare_exclude2.append('run_number')

    temp1 = param_df.drop(compare_exclude2, axis=1, errors='ignore')
    temp2 = new_row.drop(compare_exclude, axis=1, errors='ignore')
    temp_merge = pd.merge(temp1, temp2)
    # This is needed to ensure proper order in some cases (if param_dict has less items than the table has
    #   columns)
    column_labels = list(temp_merge.columns)
    column_labels.append('run_number')
    run_number = temp_merge.shape[0]

    if run_number == 0:
        return False

    return True

    # if not os.path.exists(table_path):
    #     return False
    # full_df = pd.read_csv(table_path, index_col=0)
    # param_dict_df = pd.DataFrame(param_dict, index=[0])
    # temp1 = full_df.drop(['run_time'], axis=1, errors='ignore')
    # temp1 = temp1.drop(['run_number'], axis=1, errors='ignore')
    # temp1 = temp1.drop(compare_exclude, axis=1, errors='ignore')
    # temp2 = param_dict_df.drop(['run_time'], axis=1, errors='ignore')
    # temp2 = temp2.drop(compare_exclude, axis=1, errors='ignore')
    # merged_df = temp1.reset_index().merge(temp2).set_index('index')
    # if merged_df.shape[0] == 0 or set(temp1.keys()) != set(temp2.keys()):
    #     if verbose:
    #         print("Error: run matching parameters {} not found".format(param_dict))
    #     return False
    # else:
    #     return True

def load_from_id(run_name, run_id, table_path='output/param_table.csv'):  # Todo: get it working with more filetypes
    """
    Given the name of the run, the ID of the run, and the directory of the output table, load the data.

    Args:
        run_name ():
        run_id ():
        table_dir (): Name for the directory of the table. Cannot be inside another directory other than the current
            working one.

    Returns:

    """
    # md = io.loadmat(basedir + 'output/' + str(run_id) + '/collected_data.mat')
    # md = pkl.load(open(basedir + output_dir + '/' + run_name + '_' + str(run_id) + '/output.pkl', 'rb'))
    # params = io.loadmat(basedir + 'output/' + str(run_id) + '/PARAMS.mat')
    # params = pkl.load(open(basedir + output_dir + '/' + run_name + '_' + str(run_id) + '/params.pkl', 'rb'))
    table_dir = '/'.join(table_path.split('/')[:-1])
    try:
        hf = h5py.File(table_dir + '/' + run_name + '_' + str(run_id) + '/' + data_file_name + '.h5', 'r')
    except OSError:
        hf = h5py.File(table_dir + '/' + run_name + '_' + str(run_id) + '/' + data_file_name + '.hdf5', 'r')

    output = hf['output']
    params = hf['parameters']
    return output, params

def load_data(param_dict, run_name, table_path='output/param_table.csv', ret_as_dict=True):
    """

    Args:
        param_dict (dict): Dictionary of parameters of interest (doesn't need to be comprehensive, but
        should uniquely determine the run).

    Returns:
        output: output data that has been collected
        params: parameters for the run
        run_id:
            TODO: put more info here
        nonunique_params (dict): Dictionary of parameters that have non-unique values.

    Exceptions:
        If the parameters in param_dict don't uniquely determine the run, then an error message will be
        output to say this. The function will then return nonunique_params.

    """

    # table_dir = table_dir + '/' + 'param_table.csv'
    # table_
    table_dir = '/'.join(table_path.split('/')[:-1])
    full_df = pd.read_csv(table_path, index_col=0)
    param_dict_df = pd.DataFrame(param_dict, index=[0])
    temp1 = full_df.drop(['run_time'], axis=1, errors='ignore')
    merged_df = temp1.reset_index().merge(param_dict_df).set_index('index')
    # run_name = param_dict['run_name']
    if merged_df.shape[0] == 1:
        output, params = load_from_id(run_name, merged_df.index[0], table_path=table_path)
        if ret_as_dict:
            output = hdf5group_to_dictionary(output)
            params = hdf5group_to_dictionary(params)
        run_id = merged_df.index[0]
        run_dir = table_dir + '/' + run_name + '_' + str(run_id)
        return output, params, run_id, run_dir
    elif merged_df.shape[0] > 1:
        nonunique_params = {}
        for cind in merged_df.columns:
            c = merged_df[cind].values
            cd = set(c)
            if len(cd) > 1:
                nonunique_params[cind] = cd
        str1 = "The parameters in param_dict don't uniquely determine the run."
        str2 = "Here are the nonunique parameters: {}".format(nonunique_params)
        raise KeyError(str1 + str2)
    elif merged_df.shape[0] == 0:
        raise KeyError("Error: run matching parameters {} not found".format(param_dict))

# %% Untested

def update_table(output_dir='output'):
    basedir = get_base_dir()
    out_dir = basedir + '/' + output_dir
    run_ids = [name[-1] for name in os.listdir(out_dir) if os.path.isdir(out_dir + '/' + name)]
    run_names = [name[:-2] for name in os.listdir(out_dir) if os.path.isdir(out_dir + '/' + name)]
    table_dir = './' + out_dir + '/param_table.csv'
    param_df = pd.DataFrame()
    for it, run_id in enumerate(run_ids):
        name = run_names[it]
        with h5py.File(out_dir + '/' + name + '_' + run_id + '/data.hdf5', 'r') as hf:  # Todo: resolve hdf5 vs h5 ext
            tbl_params = hdf5group_to_dictionary(hf['table_parameters'])
        new_row = pd.DataFrame(tbl_params, index=[int(run_id)])
        if it == 0:
            param_df = new_row.copy()
            param_df['run_number'] = 0
        elif it > 0:
            temp1 = param_df.drop(['ic_seed', 'run_number', 'run_time'], axis=1)
            temp2 = new_row.drop(['ic_seed', 'run_time'], axis=1)
            run_number = pd.merge(temp1, temp2).shape[0]
            new_row['run_number'] = run_number
            param_df = param_df.append(new_row)

    param_df.to_csv(table_dir)

def delete_from_id(run_name, run_id, table_path='output/param_table.csv'):
    import shutil

    # table_dir = output_dir + '/' + 'param_table.csv'
    table_dir = table_path.split('/')[0]
    data_dir = table_dir + '/' + run_name + '_' + str(run_id)

    shutil.rmtree(data_dir)

    full_df = pd.read_csv(table_dir, index_col=0)
    full_df = full_df.drop(run_id)
    full_df.to_csv(table_dir)
