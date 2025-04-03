"""
This module provides functionality for pipe characteristics.

Author: Krit Tangsongcharoen
Date: 2025-04-03
"""

from dataclasses import dataclass

import numpy as np

IArray2D = np.ndarray  # Int numpy array 2D
FArray = np.ndarray  # Float numpy array

# Constant setting
DENSITY = 700.0  # unit: kg/m^3
GRAVITATIONAL_ACCEL = 9.81  # unit: m/s^2
k = 2.0  # empirical parameter


@dataclass
class SummarizeResult1:
    """
    Object to match "key" and "value: (pressures, flows)"
    There are 4 keys: sources, junctions, sinks, and all
    """

    def __init__(self: object, **kwargs: tuple[FArray, FArray]) -> None:
        for key, (pressure_value, flow_value) in kwargs.items():
            setattr(
                self,
                key,
                self.SubObject(pressures=pressure_value, flows=flow_value),
            )

    class SubObject:
        """
        Suboject to split pressures and flows to .pressures and
        .flows respectively
        """

        def __init__(self: object, **kwargs: tuple[FArray, FArray]) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)


def calculate_flow_from_sources(
    flow_sources: FArray,  # Shape (number of sources,)
    num_vertices: int,  # the number of sources
    edge_indices: IArray2D,  # Shape (number of edges, 2)
) -> FArray:  # Shape (number of total nodes,)
    """
    calculate the flows of every junctions and sinks from continuity equation
        f_j = summation fij over i; i is the init node which has the target
        at jth node

    Args:
        flow_sources (FArray)   : set of the mass flow rate of each pipe in
                                    kilogram per second.
        num_vertices (int)      : the number of vertices (aka. nodes)
        edge_indices (LArray2D) : indices of every init nodes and paired target
                                    of that flow,
                                    ordering [sources, junctions, sinks]
                                    e.g. node 1 -> 3, and 2 -> 3, edge_indices
                                    = np.array[[1, 2], [3, 3]]

    Returns:
        FArray  : set of each of flow rate in kilogram per second.
    """

    # Determine the number of sources
    num_sources = flow_sources.size

    # Node indices which aren't sources
    node_idx_not_sources = np.arange(num_sources, num_vertices)

    # Assign the sources flows
    flows = np.zeros(num_vertices)
    flows[:num_sources] = flow_sources

    # Calculate the rest flows
    for idx in node_idx_not_sources:
        ind_target = edge_indices[0][np.where(edge_indices[1] == idx)]
        flows[idx] = (flows[ind_target]).sum()

    return flows


def calculate_p_flow(
    edge_indices: IArray2D,  # Shape (number of edges, 2)
    heights: FArray,  # Shape (number of total nodes,)
    bs: FArray,  # Shape (number of edges,)
    p_sinks: FArray,  # Shape (number of sinks,)
    flow_sources: FArray,  # Shape (number of sources)
) -> (
    SummarizeResult1
):  # Shape pressures: (number of total nodes,), flows:(number of total nodes,)
    """
    Calculate the flows of every junctions and sinks
    from continuity equation
        f_j = summation fij over i

    and also calculate the pressure of sources and
    junctions from the pressures of sinks according
    to govern equation
        P_j = P_i - rho g (h_j - h_i) + Bij * fij ^ k



    Args:
        edge_indices (LArray2D) : indices of every init nodes and pair target
                                    of that every flow
                                    e.g. node 1 -> 3, and 2 -> 3, edge_indices
                                    = np.array[[1, 2], [3, 3]]
        heights (FArray)        : set of the height of each node in meters.
        bs (FArray)             : set of the pipe Characteristic of each flow
                                    in Pascal second^k per kilogram^k.
        p_sinks (FArray)        : set of the pressure of sinks in Pascal units.
        flow_sources (FArray)   : set of the mass flow rate of each pipe in
                                    kilogram per second.

    Constant args:
        DENSITY (float)             : DENSITY of fluid in kilogram per cubic
                                        meters.
        GRAVITATIONAL_ACCEL (float) : gravitational acceleration in meter per
                                        square seconds.
        k (float)                   : constant exponential of flow to determine
                                        the effect to friction (dimensionless
                                        variable).

    Returns:
        SummarizeResult1   : a object that collects a set of every node's
                                pressure and a set of every pipe's flow
                                in the unit of Pascal and kilogram per second
                                respectively.

    """

    # Determine the numbers of sources, vertices, sinks, and edges
    num_sources = flow_sources.size
    num_vertices = heights.size
    num_sinks = p_sinks.size
    num_edges = edge_indices[0].size

    flows = calculate_flow_from_sources(flow_sources,
                                        num_vertices,
                                        edge_indices,
                                        )

    # Assign the sinks' pressure
    ps = np.zeros(num_vertices)
    ps[-num_sinks:] = p_sinks

    # Calculate the rest pressures
    for edge_idx in np.arange(num_edges - 1, -1, -1):
        idx_init = edge_indices[0][edge_idx]
        idx_target = edge_indices[1][edge_idx]
        diff_height = heights[idx_target] - heights[idx_init]
        # P_init = P_target + rho g (h_target - h_init) - Bij * fij ^ k
        ps[idx_init] = (
            ps[idx_target]
            + DENSITY * GRAVITATIONAL_ACCEL * (diff_height)
            - bs[edge_idx] * (flows[idx_init]) ** k
        )

    return SummarizeResult1(
        sources=(ps[:num_sources], flows[:num_sources]),
        junctions=(
            ps[num_sources:num_vertices - num_sinks],
            flows[num_sources:num_vertices - num_sinks],
        ),
        sinks=(ps[num_vertices - num_sinks:], flows[num_vertices - num_sinks]),
        all=(ps, flows),
    )


@dataclass
class SummarizeResult2:
    """
    Object to match "key" and "value: pipe characteristic"
    There are 4 keys: sources, junctions, sinks, and all
    """

    def __init__(self: object, **kwargs: tuple[FArray, FArray]) -> None:
        for key, b_value in kwargs.items():
            setattr(self, key, b_value)


def calculate_pipe_characteristic(
    edge_indices: IArray2D,  # Shape (number of edges, 2)
    heights: FArray,  # Shape (number of total nodes,)
    p_s: FArray,  # Shape (number of total nodes,)
    flow_sources: FArray,  # Shape (number of sources)
    num_sinks: int,  # the number of sinks
) -> SummarizeResult2:  # (number of total nodes,), (number of total nodes,)
    """
    Calculate the flows of every junctions and sinks from continuity equation
        f_j = summation fi,j over i

    and then solving the pipe characteristics from govern equation
        Bij = 1/fij ^ k (P_j - P_i - rho g (h_i - h_j)
        where i is the init node, and j is the targeted node.



    Args:
        edge_indices (LArray2D) : indices of every init nodes and pair target
                                    of that every flow
                                    e.g. node 1 -> 3, and 2 -> 3, edge_indices
                                    = np.array[[1, 2], [3, 3]]
        heights (FArray)        : set of the height of each node in meters.
        P_s (FArray)            : set of the pressure of each node in Pascal
                                    units.
        flow_sources (FArray)   : set of the mass flow rate of each pipe in
                                    kilogram per second.

    Constant args:
        DENSITY (float)             : DENSITY of fluid in kilogram per cubic
                                        meters.
        GRAVITATIONAL_ACCEL (float) : gravitational acceleration in meter per
                                        square seconds.
        k (float)                   : constant exponential of flow to determine
                                        the effect to friction
                                        (dimensionless variable).

    Returns:
        SummarizeResult_2   : each pipe's characteristic in Pascal second per
                                kilogram.
    """

    # Determine the numbers of sources, vertices, sinks, and edges
    num_sources = flow_sources.size
    num_vertices = heights.size
    num_edges = edge_indices[0].size

    # Calculate the rest flows
    flows = calculate_flow_from_sources(flow_sources,
                                        num_vertices,
                                        edge_indices,)

    # Solving for Pipe Characteristics
    bs = np.zeros(num_edges)
    for edge_idx in np.arange(num_edges):
        idx_init = edge_indices[0][edge_idx]
        idx_target = edge_indices[1][edge_idx]
        diff_height = heights[idx_target] - heights[idx_init]
        # Bij = 1/fij ^ k (P_target - P_init - rho g (h_init - h_target)
        bs[edge_idx] = (
            p_s[idx_target]
            - p_s[idx_init]
            - DENSITY * GRAVITATIONAL_ACCEL * (diff_height)
        ) / (flows[idx_init]) ** k

    return SummarizeResult2(
        sources=bs[:num_sources],
        junctions=bs[num_sources:num_vertices - num_sinks],
        sinks=bs[num_vertices - num_sinks:],
        all=bs,
    )
