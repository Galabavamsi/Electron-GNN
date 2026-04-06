import numpy as np
import plotly.graph_objects as go
import pandas as pd
import os

def load_xyz_and_grid(xyz_path):
    """
    Parse ReSpect .xyz file to get Atoms and Grid 3D coords.
    """
    atoms = []
    atom_positions = []
    
    grid_points = []
    
    in_atoms = False
    in_grid = False
    
    with open(xyz_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("[Atoms]"):
                in_atoms = True
                continue
            if line.startswith("[Grid]"):
                in_atoms = False
                in_grid = True
                continue
                
            if in_atoms and line:
                parts = line.split()
                if len(parts) >= 6:
                    sym = parts[0]
                    x = float(parts[3])
                    y = float(parts[4])
                    z = float(parts[5])
                    atoms.append(sym)
                    atom_positions.append([x, y, z])
                    
            if in_grid and line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        grid_points.append([x, y, z])
                    except ValueError:
                        pass
                        
    return atoms, np.array(atom_positions), np.array(grid_points)

def load_density_file(rho_path):
    """
    Load a ReSpect density file (e.g., .rho.00000).
    """
    densities = []
    with open(rho_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    densities.append(float(parts[1]))
                except ValueError:
                    pass
    return np.array(densities)

def plot_molecule_heatmap_3d(atoms, atom_positions, grid_points, density_diff, threshold=1e-5):
    """
    Given atom coords and density difference upon the grid, generate a Plotly 3D Figure.
    """
    fig = go.Figure()
    
    # 1. Plot Atoms
    color_map = {'N': '#1f77b4', 'H': '#d3d3d3', 'C': '#7f7f7f', 'O': '#d62728'}
    sizes = {'N': 25, 'H': 12, 'C': 20, 'O': 22}
    
    # Draw Bonds (distance based)
    bond_dist = 2.5 # ~1.3 Angstroms in atomic units
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            dist = np.linalg.norm(atom_positions[i] - atom_positions[j])
            if dist < bond_dist:
                fig.add_trace(go.Scatter3d(
                    x=[atom_positions[i, 0], atom_positions[j, 0]],
                    y=[atom_positions[i, 1], atom_positions[j, 1]],
                    z=[atom_positions[i, 2], atom_positions[j, 2]],
                    mode='lines',
                    line=dict(color='gray', width=6),
                    hoverinfo='skip',
                    showlegend=False
                ))

    for i, sym in enumerate(atoms):
        fig.add_trace(go.Scatter3d(
            x=[atom_positions[i, 0]],
            y=[atom_positions[i, 1]],
            z=[atom_positions[i, 2]],
            mode='markers+text',
            marker=dict(
                size=sizes.get(sym, 15),
                color=color_map.get(sym, 'green'),
                line=dict(width=2, color='DarkSlateGrey'),
                symbol='circle'
            ),
            text=[f"<b>{sym}</b>"],
            textfont=dict(size=14, color="black"),
            textposition="middle center", # Inside the atom
            name=f"Atom {sym}",
            showlegend=False
        ))
        
    # 2. Render Electron Density Cloud using Semi-Transparent point clouds
    mask = np.abs(density_diff) > threshold
    if np.any(mask):
        sub_grid = grid_points[mask]
        sub_density = density_diff[mask]
        
        # Sort by absolute density to render highest density last (on top)
        sort_idx = np.argsort(np.abs(sub_density))
        sub_grid = sub_grid[sort_idx]
        sub_density = sub_density[sort_idx]
        
        vmax = np.max(np.abs(sub_density))
        
        fig.add_trace(go.Scatter3d(
            x=sub_grid[:, 0],
            y=sub_grid[:, 1],
            z=sub_grid[:, 2],
            mode='markers',
            marker=dict(
                size=8, # Large size to blur together like a cloud
                color=sub_density, 
                colorscale='RdBu', # Standard chemistry scale: Red / Blue
                cmin=-vmax,
                cmax=vmax,
                opacity=0.25, # High transparency
                symbol='circle',
                line=dict(width=0), # No hard edges
                colorbar=dict(
                    title="Volume Δρ", 
                    x=1.0, # Push slightly right
                    len=0.8,
                    thickness=20
                )
            ),
            name="Density Cloud",
            hoverinfo='skip',
            showlegend=False
        ))
        
    # Standardize axes to be clean, invisible grids for better molecular visibility
    limit = 4.5
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-limit, limit], showgrid=False, zeroline=False, showbackground=False, visible=False),
            yaxis=dict(range=[-limit, limit], showgrid=False, zeroline=False, showbackground=False, visible=False),
            zaxis=dict(range=[-limit, limit], showgrid=False, zeroline=False, showbackground=False, visible=False),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2) # Better isometric default angle
            )
        ),
        margin=dict(r=0, l=0, b=0, t=30),
        title="<b>Dynamic 3D Electron Density Fluctuations</b>",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    return fig

def plot_molecule_heatmap_3d_animation(atoms, atom_positions, grid_points, frames_data, threshold=1e-5):
    """
    Generate a Plotly Figure with native JS animation across timeframes.
    frames_data: list of tuples (time_label, density_diff_array)
    """
    fig = go.Figure()
    
    color_map = {'N': '#1f77b4', 'H': '#d3d3d3', 'C': '#7f7f7f', 'O': '#d62728'}
    sizes = {'N': 25, 'H': 12, 'C': 20, 'O': 22}
    
    # 1. Draw Bonds
    bond_dist = 2.5
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            dist = np.linalg.norm(atom_positions[i] - atom_positions[j])
            if dist < bond_dist:
                fig.add_trace(go.Scatter3d(
                    x=[atom_positions[i, 0], atom_positions[j, 0]],
                    y=[atom_positions[i, 1], atom_positions[j, 1]],
                    z=[atom_positions[i, 2], atom_positions[j, 2]],
                    mode='lines',
                    line=dict(color='gray', width=6),
                    hoverinfo='skip',
                    showlegend=False
                ))

    # 2. Draw Atoms
    for i, sym in enumerate(atoms):
        fig.add_trace(go.Scatter3d(
            x=[atom_positions[i, 0]],
            y=[atom_positions[i, 1]],
            z=[atom_positions[i, 2]],
            mode='markers+text',
            marker=dict(
                size=sizes.get(sym, 15),
                color=color_map.get(sym, 'green'),
                line=dict(width=2, color='DarkSlateGrey'),
                symbol='circle'
            ),
            text=[f"<b>{sym}</b>"],
            textfont=dict(size=14, color="black"),
            textposition="middle center",
            name=f"Atom {sym}",
            showlegend=False
        ))
        
    # 3. Add initial frame density trace
    t0_label, d0_rho = frames_data[0]
    mask = np.abs(d0_rho) > threshold
    if np.any(mask):
        sg = grid_points[mask]
        sd = d0_rho[mask]
        si = np.argsort(np.abs(sd))
        sg, sd = sg[si], sd[si]
        vmax = np.max(np.abs(sd))
    else:
        sg = np.zeros((0, 3))
        sd = np.zeros(0)
        vmax = 1e-5
        
    trace_idx = len(fig.data) # This trace will be targeted by the frames
    
    fig.add_trace(go.Scatter3d(
        x=sg[:, 0] if len(sg)>0 else [None],
        y=sg[:, 1] if len(sg)>0 else [None],
        z=sg[:, 2] if len(sg)>0 else [None],
        mode='markers',
        marker=dict(
            size=8,
            color=sd if len(sd)>0 else [0],
            colorscale='RdBu',
            cmin=-vmax, cmax=vmax,
            opacity=0.25,
            symbol='circle',
            line=dict(width=0),
            colorbar=dict(title="Volume Δρ", x=1.0, len=0.8, thickness=20)
        ),
        name="Density Cloud",
        hoverinfo='skip',
        showlegend=False
    ))
    
    # 4. Build frames array to animate the density cloud
    frames = []
    for t_label, d_rho in frames_data:
        m = np.abs(d_rho) > threshold
        if np.any(m):
            sg_f = grid_points[m]
            sd_f = d_rho[m]
            si_f = np.argsort(np.abs(sd_f))
            sg_f, sd_f = sg_f[si_f], sd_f[si_f]
            v_m = np.max(np.abs(sd_f))
        else:
            sg_f = np.zeros((0, 3))
            sd_f = np.zeros(0)
            v_m = 1e-5
            
        frame_trace = go.Scatter3d(
            x=sg_f[:, 0] if len(sg_f)>0 else [None],
            y=sg_f[:, 1] if len(sg_f)>0 else [None],
            z=sg_f[:, 2] if len(sg_f)>0 else [None],
            marker=dict(
                size=8,
                color=sd_f if len(sd_f)>0 else [0],
                cmin=-v_m, cmax=v_m
            )
        )
        # Point to the specific trace_idx to update just the volumetric cloud
        frames.append(go.Frame(data=[frame_trace], name=f"{t_label}", traces=[trace_idx]))
        
    fig.frames = frames
    
    # 5. Add Play/Pause GUI and slider
    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=0, x=0.1, xanchor="right", yanchor="top",
            pad=dict(t=80, r=10),
            direction="left",
            buttons=[
                dict(label="▶ Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True, transition=dict(duration=0))]),
                dict(label="⏸ Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
            ]
        )],
        sliders=[dict(
            active=0,
            yanchor="top", xanchor="left",
            currentvalue=dict(font=dict(size=14), prefix="Time (a.u.): ", visible=True, xanchor="right"),
            transition=dict(duration=0), # Transition 0 prevents interpolation bugs on discrete geometry
            pad=dict(b=10, t=50),
            len=0.9, x=0.1, y=0,
            steps=[dict(args=[[f"{t_label}"], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))],
                        label=str(t_label), method="animate") for t_label, _ in frames_data]
        )]
    )
    
    # Standardize axes limits
    limit = 4.5
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-limit, limit], showgrid=False, zeroline=False, showbackground=False, visible=False),
            yaxis=dict(range=[-limit, limit], showgrid=False, zeroline=False, showbackground=False, visible=False),
            zaxis=dict(range=[-limit, limit], showgrid=False, zeroline=False, showbackground=False, visible=False),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
        ),
        margin=dict(r=0, l=0, b=0, t=30),
        title="<b>Animated 3D Electron Density Fluctuations $\Delta\\rho(r)$</b>",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig
