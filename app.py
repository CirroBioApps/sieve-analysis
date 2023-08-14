#!/usr/bin/env streamlit

from time import sleep
from typing import List
from cirro import DataPortal
from cirro.api.clients.portal import DataPortalClient
from cirro.sdk.exceptions import DataPortalAssetNotFound
import io
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from streamlit.runtime.scriptrunner import script_run_context
from streamlit.runtime.scriptrunner import get_script_run_ctx
import threading


def session_cache(func):
    def inner(*args, **kwargs):

        # Get the session context, which has a unique ID element
        ctx = get_script_run_ctx()

        # Define a cache key based on the function name and arguments
        cache_key = ".".join([
            str(ctx.session_id),
            func.__name__,
            ".".join(map(str, args)),
            ".".join([
                f"{k}={v}"
                for k, v in kwargs.items()
            ])
        ])

        # If the value has not been computed
        if st.session_state.get(cache_key) is None:
            # Compute it
            st.session_state[cache_key] = func(
                *args,
                **kwargs
            )

        # Return that value
        return st.session_state[cache_key]
    
    return inner


def cirro_login(login_empty):
    # If we have not logged in yet
    if st.session_state.get('DataPortal') is None:

        # Connect to Cirro - capturing the login URL
        auth_io = io.StringIO()
        cirro_login_thread = threading.Thread(
            target=cirro_login_sub,
            args=(auth_io,)
        )
        script_run_context.add_script_run_ctx(cirro_login_thread)

        cirro_login_thread.start()

        login_string = auth_io.getvalue()

        while len(login_string) == 0 and cirro_login_thread.is_alive():
            sleep(5)
            login_string = auth_io.getvalue()

        login_empty.write(login_string)
        cirro_login_thread.join()

    msg = "Error: Could not log in to Cirro"
    assert st.session_state.get('DataPortal') is not None, msg


def cirro_login_sub(auth_io: io.StringIO):

    st.session_state['DataPortal-client'] = DataPortalClient(auth_io=auth_io)
    st.session_state['DataPortal'] = DataPortal(client=st.session_state['DataPortal-client'])


def list_datasets_in_project(project_name):

    # Connect to Cirro
    portal = st.session_state['DataPortal']

    # Access the project
    project = portal.get_project_by_name(project_name)

    # Get the list of datasets available (using their easily-readable names)
    return [""] + [ds.name for ds in project.list_datasets()]


@session_cache
def list_projects() -> List[str]:

    # Connect to Cirro
    portal = st.session_state['DataPortal']

    # List the projects available
    project_list = portal.list_projects()

    # Return the list of projects available (using their easily-readable names)
    return [proj.name for proj in project_list]


def prompt_dataset_in_project(project_name):
    """Prompt the user to select a dataset"""

    # Get the list of datasets available in the project
    all_datasets = list_datasets_in_project(project_name)

    # Prompt the user to select one of those datasets
    return st.sidebar.selectbox(
        "Dataset:",
        all_datasets
    )


def prompt_project():
    """Prompt the user to select a project from Cirro."""

    # Get the list of datasets available in the project
    project_list = list_projects()

    # Prompt the user to select one of those datasets
    return st.sidebar.selectbox(
        "Project:",
        [""] + project_list
    )


@session_cache
def get_dataset(project_name, dataset_name):
    """Return a Cirro Dataset object."""

    # Connect to Cirro
    portal = st.session_state['DataPortal']

    # Access the project
    project = portal.get_project_by_name(project_name)

    # Get the dataset
    return project.get_dataset_by_name(dataset_name)


@session_cache
def read_csv(project_name, dataset_name, fn, **kwargs):
    """Read a CSV from a dataset in Cirro."""

    return (
        get_dataset(project_name, dataset_name)
        .list_files()
        .get_by_name(f"data/{fn}")
        .read_csv(**kwargs)
    )


@session_cache
def readlines(project_name, dataset_name, fn, **kwargs):
    """Read a CSV from a dataset in Cirro."""

    return (
        get_dataset(project_name, dataset_name)
        .list_files()
        .get_by_name(f"data/{fn}")
        .readlines()
    )


def filter_metadata(metadata):
    # Make a copy of the metadata table
    filtered_metadata = metadata.copy()

    # Iterate over each column
    for cname, cvals in metadata.items():

        # If any of the values in the column are strings
        if cvals.apply(lambda v: isinstance(v, str)).any():

            # Let the user select which unique values to keep
            keep_values = st.sidebar.multiselect(
                cname,
                cvals.unique(),
                default=cvals.unique()
            )

            # Apply the filter
            if len(keep_values) < cvals.unique().shape[0]:
                st.write(
                    f"Filtering to {cname} in [{' | '.join(keep_values)}]"
                )
                filtered_metadata = filtered_metadata.loc[
                    filtered_metadata[cname].isin(keep_values)
                ]

        # Otherwise the values are all non-strings (assume numeric)
        else:

            # Get the range of values
            minval = cvals.min()
            maxval = cvals.max()

            # Show the user a slider
            filtered_minval, filtered_maxval = st.sidebar.slider(
                cname,
                minval,
                maxval,
                (minval, maxval)
            )

            # Use the selected values to filter the table
            if filtered_minval > minval:
                st.write(f"Filtering to {cname} >= {filtered_minval}")
                filtered_metadata = filtered_metadata.loc[
                    filtered_metadata[cname] >= filtered_minval
                ]
            if filtered_maxval > maxval:
                st.write(f"Filtering to {cname} <= {filtered_maxval}")
                filtered_metadata = filtered_metadata.loc[
                    filtered_metadata[cname] <= filtered_maxval
                ]

    return filtered_metadata


@session_cache
def read_aligned_data(project_name, dataset_name) -> pd.DataFrame:

    # Read the aligned FASTA sequences into a DataFrame
    # using 1-based coordinates
    obs = read_fasta(
        project_name,
        dataset_name,
        "env_aa_obs.fasta"
    )

    # Read in the reference sequences too
    ref = read_fasta(
        project_name,
        dataset_name,
        "env_aa_ref.fasta"
    )

    # Get the mapping of alignment positions to reference positions
    env_map = read_csv(
        project_name,
        dataset_name,
        "env.map",
        sep="|",
        index_col=0
    )

    # Merge the data together, adding env_map as columns
    return pd.concat([
        obs.assign(datatype="obs"),
        ref.assign(datatype="ref")
    ]).merge(
        env_map,
        how='outer',
        left_on="aln_pos",
        right_index=True
    )


def read_fasta(project_name, dataset_name, file_name):
    """
    Input Format:
    >seq1
    MRAKEM
    >seq2
    MRAKEM

    Output Format (DataFrame):

    | aln_pos | header | aa |
    | 1       | seq1   | M  |
    | 2       | seq1   | R  |
    | 3       | seq1   | A  |
    | 4       | seq1   | K  |
    ...

    """

    # Read in the FASTA as a list of lines
    fasta_lines = readlines(project_name, dataset_name, file_name)

    dat = dict()
    header = None
    for line in fasta_lines:
        if line.startswith(">"):
            header = line[1:]
        else:
            assert header is not None, "FASTA must start with header line"
            dat[header] = dat.get(header, "") + line.strip()

    return pd.DataFrame([
        dict(
            aln_pos=ix+1,
            header=header,
            aa=aa
        )
        for header, seq in dat.items()
        for ix, aa in enumerate(seq)
    ])


def app():

    # Set up a container for the login text
    login_empty = st.empty()
    # Print the URL print in that area, if needed
    cirro_login(login_empty)
    # Clear the display after the login is complete
    login_empty.empty()

    # Select a dataset from the 'BBE Sieve Analysis' project in Cirro
    project_name = prompt_project()

    # Wait until a valid project is selected
    if project_name == "":
        st.write("Please select a Cirro project from the menu on the left")
        return

    dataset_name = prompt_dataset_in_project(project_name)

    # If no dataset was selected, prompt the user
    if dataset_name == "":
        st.write("Please select a Cirro dataset")
        return

    # Print a header at the top of the page showing the dataset
    st.write(f"### {dataset_name}")

    # Read the table of gene coordinates
    try:
        regions = read_csv(
            project_name,
            dataset_name,
            "env_locations.csv",
            index_col=0
        )
    except DataPortalAssetNotFound:
        st.write("Expected files not found in this dataset")
        return
    # st.write(f"Read in {regions.shape[0]} records from env_locations.csv")

    # Provide the user with the ability to select a region of interest
    selected_region = st.sidebar.selectbox(
        "Gene / Region",
        regions.index.values
    )

    # Read the metadata table from the dataset
    metadata = read_csv(
        project_name,
        dataset_name,
        "vtn505_export_cirro.csv",
        index_col=0
    )

    # Provide the user with options to filter the metadata
    filtered_metadata = filter_metadata(metadata)

    # Get the aligned data in this dataset
    aligned_data = read_aligned_data(project_name, dataset_name)

    # Filter down to the samples selected by the user on the basis of metadata
    filtered_data = filter_data(aligned_data, filtered_metadata)

    # Remove any position which does not exist in the reference
    filtered_data = filtered_data.query("hxb2aa != '-'")

    # Filter to the region of interest
    start_aa = regions.loc[selected_region, "start.aa"]
    stop_aa = regions.loc[selected_region, "stop.aa"]
    filtered_data = filtered_data.loc[
        filtered_data["hxb2Pos"].apply(int).apply(
            lambda ref_pos: ref_pos >= start_aa and ref_pos <= stop_aa
        )
    ]

    # Display the selected subset of the data
    display_sequences(filtered_data, title=selected_region)


@session_cache
def filter_data(aligned_data, filtered_metadata):
    """
    Filter down to the samples selected by the user on the basis of metadata
    """

    # Keep the samples which are in the filtered_metadata table index
    filtered_pubids = list(map(str, filtered_metadata.index.values))

    # Keep the alignments for data which is either (a) from the reference or
    # (b) in the set of filtered samples
    return aligned_data.assign(
        toKeep=aligned_data.apply(
            lambda r: r['datatype'] == "ref" or r["header"].split("_")[0] in filtered_pubids, # noqa
            axis=1
        )
    ).query("toKeep").drop(columns=["toKeep"])


@session_cache
def make_wide_data(data):

    wide_data = data.pivot(
        index="header",
        columns="hxb2Pos",
        values='aa'
    ).rename(
        columns=lambda pos: int(pos)
    ).sort_index(
        axis=1
    )

    return wide_data


def display_sequences(data, title=None):
    """Display the selected metadata to the user."""

    # Make the data wide-form
    wide_data = make_wide_data(data)

    # Set up the figure
    fig = make_subplots(
        cols=1,
        rows=2,
        row_heights=[0.2, 0.8],
        shared_xaxes=True,
        vertical_spacing=0.05
    )

    # Plot the alignment as a heatmap
    fig.add_trace(
        go.Heatmap(
            z=wide_data.applymap(aa_to_int).values,
            text=wide_data.values,
            x=wide_data.columns.values,
            y=wide_data.index.values,
            showscale=False,
            colorscale="Turbo"
        ),
        row=2, col=1
    )

    # Calculate the minor allele frequency at each site
    maf = wide_data.apply(
        lambda c: (c.shape[0] - c.value_counts().values[0]) / c.shape[0]
    )
    fig.add_trace(
        go.Bar(x=maf.index.values, y=maf.values),
        row=1, col=1
    )

    # Allow the user to control the figure height
    fig.update_layout(
        height=st.sidebar.slider(
            "Figure height",
            min_value=400,
            max_value=2000,
            value=600,
            step=10
        ),
        xaxis2_title="Amino Acid Position",
        yaxis_title="MAF"
    )

    if title is not None:
        fig.update_layout(title=dict(text=title, xanchor='center', x=0.5))

    st.plotly_chart(fig)


def aa_to_int(aa):
    return dict(
        L=1,
        I=2,
        V=3,
        G=4,
        A=5,
        P=6,
        Q=7,
        N=8,
        M=9,
        T=10,
        S=11,
        C=12,
        E=13,
        D=14,
        K=15,
        R=16,
        Y=17,
        F=18,
        W=19,
        X=20
    ).get(aa)


if __name__ == "__main__":
    app()
