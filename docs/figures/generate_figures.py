#!/usr/bin/env python3
"""Generate figures for the TSR tutorial documentation."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
from pathlib import Path

# Output directory
OUTPUT_DIR = Path(__file__).parent

# Style settings
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14


def draw_frame(ax, origin, R, scale=0.3, labels=None, alpha=1.0, linewidth=2):
    """Draw a coordinate frame (3 orthogonal arrows)."""
    colors = ['#e41a1c', '#4daf4a', '#377eb8']  # red, green, blue for x, y, z
    default_labels = ['x', 'y', 'z']
    labels = labels or default_labels

    for i, (color, label) in enumerate(zip(colors, labels)):
        direction = R[:, i] * scale
        ax.quiver(*origin, *direction, color=color, arrow_length_ratio=0.15,
                  linewidth=linewidth, alpha=alpha)
        label_pos = origin + direction * 1.2
        ax.text(*label_pos, label, color=color, fontsize=10, fontweight='bold',
                ha='center', va='center', alpha=alpha)


def fig1_tsr_concept():
    """Figure 1: High-level TSR concept - constrained region in task space."""
    fig = plt.figure(figsize=(10, 5))

    # Left: Without TSR (single pose)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Single Goal Pose', pad=10)

    # Draw table
    table_verts = [
        [(-0.5, -0.3, 0), (0.5, -0.3, 0), (0.5, 0.3, 0), (-0.5, 0.3, 0)]
    ]
    table = Poly3DCollection(table_verts, alpha=0.3, facecolor='brown', edgecolor='black')
    ax1.add_collection3d(table)

    # Draw single mug pose
    draw_frame(ax1, np.array([0, 0, 0.1]), np.eye(3), scale=0.15,
               labels=['', '', ''])

    # Draw mug (simple cylinder representation)
    theta = np.linspace(0, 2*np.pi, 30)
    z_cyl = np.linspace(0.1, 0.25, 10)
    theta_grid, z_grid = np.meshgrid(theta, z_cyl)
    r = 0.04
    x_cyl = r * np.cos(theta_grid)
    y_cyl = r * np.sin(theta_grid)
    ax1.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.6, color='#4169E1')

    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(-0.3, 0.3)
    ax1.set_zlim(0, 0.4)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=25, azim=-60)

    # Right: With TSR (region of valid poses)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('TSR: Region of Valid Poses', pad=10)

    # Draw table
    ax2.add_collection3d(Poly3DCollection(table_verts, alpha=0.3,
                                           facecolor='brown', edgecolor='black'))

    # Draw valid region (disk on table)
    theta = np.linspace(0, 2*np.pi, 50)
    r_vals = np.linspace(0, 0.15, 20)
    theta_grid, r_grid = np.meshgrid(theta, r_vals)
    x_disk = r_grid * np.cos(theta_grid)
    y_disk = r_grid * np.sin(theta_grid)
    z_disk = np.ones_like(x_disk) * 0.001
    ax2.plot_surface(x_disk, y_disk, z_disk, alpha=0.4, color='#32CD32',
                     edgecolor='none')

    # Draw multiple mug positions (ghosted)
    for pos in [(-0.08, 0.05), (0.1, -0.03), (0, 0.1), (-0.05, -0.08)]:
        theta_cyl = np.linspace(0, 2*np.pi, 20)
        z_cyl = np.linspace(0.1, 0.25, 5)
        theta_grid, z_grid = np.meshgrid(theta_cyl, z_cyl)
        r = 0.04
        x_cyl = pos[0] + r * np.cos(theta_grid)
        y_cyl = pos[1] + r * np.sin(theta_grid)
        ax2.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.25, color='#4169E1')

    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(-0.3, 0.3)
    ax2.set_zlim(0, 0.4)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_tsr_concept.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig1_tsr_concept.png")


def fig2_coordinate_frames():
    """Figure 2: TSR coordinate frames (world, TSR origin, end-effector)."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('TSR Coordinate Frames', pad=15)

    # World frame at origin
    world_origin = np.array([0, 0, 0])
    draw_frame(ax, world_origin, np.eye(3), scale=0.4,
               labels=['$x_w$', '$y_w$', '$z_w$'])
    ax.text(0, 0, -0.15, 'World Frame\n(origin)', ha='center', fontsize=10)

    # TSR frame (T0_w transform)
    tsr_origin = np.array([0.8, 0.3, 0.4])
    # Slight rotation for visual clarity
    angle = np.pi/6
    R_tsr = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    draw_frame(ax, tsr_origin, R_tsr, scale=0.3,
               labels=['$x_{tsr}$', '$y_{tsr}$', '$z_{tsr}$'])
    ax.text(0.8, 0.3, 0.2, 'TSR Frame\n($T_0^w$)', ha='center', fontsize=10)

    # Draw T0_w arrow (use quiver for 3D)
    start = np.array([0.15, 0.05, 0.05])
    end = np.array([0.7, 0.25, 0.35])
    direction = end - start
    ax.quiver(*start, *direction, color='purple', arrow_length_ratio=0.1, linewidth=2)
    ax.text(0.4, 0.15, 0.25, '$T_0^w$', color='purple', fontsize=14, fontweight='bold')

    # End-effector frame (Tw_e from TSR frame)
    ee_origin = tsr_origin + R_tsr @ np.array([0.25, 0.1, 0.15])
    R_ee = R_tsr @ np.array([
        [1, 0, 0],
        [0, np.cos(np.pi/8), -np.sin(np.pi/8)],
        [0, np.sin(np.pi/8), np.cos(np.pi/8)]
    ])
    draw_frame(ax, ee_origin, R_ee, scale=0.2,
               labels=['$x_e$', '$y_e$', '$z_e$'])
    ax.text(ee_origin[0], ee_origin[1], ee_origin[2] + 0.25,
            'End-Effector\nFrame ($T_w^e$)', ha='center', fontsize=10)

    # Draw Tw_e arrow
    mid = (tsr_origin + ee_origin) / 2
    ax.text(mid[0] + 0.1, mid[1] + 0.1, mid[2], '$T_w^e$',
            color='orange', fontsize=14, fontweight='bold')

    # Draw dashed line connecting frames
    ax.plot([tsr_origin[0], ee_origin[0]],
            [tsr_origin[1], ee_origin[1]],
            [tsr_origin[2], ee_origin[2]],
            'k--', alpha=0.5, linewidth=1.5)

    # Draw bounding box to represent Bw (constraint region)
    # This is the region where the end-effector can be relative to TSR frame
    box_size = np.array([0.3, 0.2, 0.25])
    box_center = tsr_origin + R_tsr @ np.array([0.15, 0.05, 0.1])

    # Draw translucent box
    vertices = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                v = box_center + R_tsr @ (np.array([dx, dy, dz]) * box_size / 2)
                vertices.append(v)

    # Box faces
    faces = [
        [vertices[0], vertices[1], vertices[3], vertices[2]],  # bottom
        [vertices[4], vertices[5], vertices[7], vertices[6]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[2], vertices[6], vertices[4]],  # left
        [vertices[1], vertices[3], vertices[7], vertices[5]],  # right
    ]
    box_collection = Poly3DCollection(faces, alpha=0.15, facecolor='green',
                                       edgecolor='green', linewidth=1)
    ax.add_collection3d(box_collection)
    ax.text(box_center[0], box_center[1] - 0.2, box_center[2],
            'Constraint\nRegion ($B_w$)', ha='center', fontsize=10, color='green')

    ax.set_xlim(-0.3, 1.5)
    ax.set_ylim(-0.3, 1.0)
    ax.set_zlim(-0.2, 1.0)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20, azim=-55)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_coordinate_frames.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig2_coordinate_frames.png")


def fig3_bounds_matrix():
    """Figure 3: Visualization of the 6x2 bounds matrix."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('The 6×2 Bounds Matrix $B_w$', fontsize=16, fontweight='bold')

    # Translation bounds (top row)
    titles_trans = ['X bounds: $[B_{w,0,0}, B_{w,0,1}]$',
                    'Y bounds: $[B_{w,1,0}, B_{w,1,1}]$',
                    'Z bounds: $[B_{w,2,0}, B_{w,2,1}]$']
    colors = ['#e41a1c', '#4daf4a', '#377eb8']

    for i, (ax, title, color) in enumerate(zip(axes[0], titles_trans, colors)):
        ax.set_title(title, fontsize=12)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

        # Draw the allowed range
        ax.axvspan(-0.3, 0.3, alpha=0.3, color=color)
        ax.annotate('', xy=(0.3, 0), xytext=(-0.3, 0),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=2))
        ax.text(0, 0.1, f'$B_{{w,{i},0}}$ to $B_{{w,{i},1}}$', ha='center',
                fontsize=10, color=color)

        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.3, 0.3)
        ax.set_xlabel(f'{["X", "Y", "Z"][i]} (meters)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Rotation bounds (bottom row)
    titles_rot = ['Roll bounds: $[B_{w,3,0}, B_{w,3,1}]$',
                  'Pitch bounds: $[B_{w,4,0}, B_{w,4,1}]$',
                  'Yaw bounds: $[B_{w,5,0}, B_{w,5,1}]$']

    for i, (ax, title, color) in enumerate(zip(axes[1], titles_rot, colors)):
        ax.set_title(title, fontsize=12)

        # Draw a circle representing rotation
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=1)

        # Draw the allowed angular range
        theta_range = np.linspace(-np.pi/4, np.pi/4, 50)
        ax.fill_between(np.cos(theta_range),
                        np.sin(theta_range) * 0.3,
                        np.sin(theta_range) * 1.0,
                        alpha=0.3, color=color)

        # Draw the arc
        arc_theta = np.linspace(-np.pi/4, np.pi/4, 50)
        ax.plot(0.7 * np.cos(arc_theta), 0.7 * np.sin(arc_theta),
                color=color, linewidth=3)
        ax.annotate('', xy=(0.7*np.cos(np.pi/4), 0.7*np.sin(np.pi/4)),
                    xytext=(0.7*np.cos(-np.pi/4), 0.7*np.sin(-np.pi/4)),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=2,
                                   connectionstyle='arc3,rad=0.3'))

        ax.text(0, -0.2, f'{["Roll", "Pitch", "Yaw"][i]}\n(radians)',
                ha='center', fontsize=10)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_bounds_matrix.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig3_bounds_matrix.png")


def fig4_primitives():
    """Figure 4: Geometric primitives as TSR constraint regions."""
    fig = plt.figure(figsize=(15, 10))

    primitives = [
        ('Point', 'point'),
        ('Line', 'line'),
        ('Plane', 'plane'),
        ('Box', 'box'),
        ('Ring', 'ring'),
        ('Disk', 'disk'),
        ('Cylinder', 'cylinder'),
        ('Shell', 'shell'),
        ('Sphere', 'sphere'),
    ]

    for idx, (name, ptype) in enumerate(primitives):
        ax = fig.add_subplot(3, 3, idx + 1, projection='3d')
        ax.set_title(name, fontsize=12, fontweight='bold')

        if ptype == 'point':
            ax.scatter([0], [0], [0], s=100, c='blue', marker='o')

        elif ptype == 'line':
            z = np.linspace(-0.5, 0.5, 50)
            ax.plot([0]*50, [0]*50, z, 'b-', linewidth=3)

        elif ptype == 'plane':
            x = np.linspace(-0.5, 0.5, 10)
            y = np.linspace(-0.5, 0.5, 10)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            ax.plot_surface(X, Y, Z, alpha=0.5, color='blue')

        elif ptype == 'box':
            # Draw box edges
            r = 0.3
            for s in [-1, 1]:
                for t in [-1, 1]:
                    ax.plot([-r, r], [s*r, s*r], [t*r, t*r], 'b-', linewidth=1)
                    ax.plot([s*r, s*r], [-r, r], [t*r, t*r], 'b-', linewidth=1)
                    ax.plot([s*r, s*r], [t*r, t*r], [-r, r], 'b-', linewidth=1)
            # Draw transparent faces
            verts = [
                [[-r,-r,-r], [r,-r,-r], [r,r,-r], [-r,r,-r]],
                [[-r,-r,r], [r,-r,r], [r,r,r], [-r,r,r]],
            ]
            ax.add_collection3d(Poly3DCollection(verts, alpha=0.2, facecolor='blue'))

        elif ptype == 'ring':
            theta = np.linspace(0, 2*np.pi, 100)
            r = 0.4
            ax.plot(r*np.cos(theta), r*np.sin(theta), [0]*100, 'b-', linewidth=3)

        elif ptype == 'disk':
            theta = np.linspace(0, 2*np.pi, 50)
            r_vals = np.linspace(0, 0.4, 20)
            T, R = np.meshgrid(theta, r_vals)
            X = R * np.cos(T)
            Y = R * np.sin(T)
            Z = np.zeros_like(X)
            ax.plot_surface(X, Y, Z, alpha=0.5, color='blue')

        elif ptype == 'cylinder':
            theta = np.linspace(0, 2*np.pi, 30)
            z = np.linspace(-0.4, 0.4, 20)
            T, Z_grid = np.meshgrid(theta, z)
            r = 0.3
            X = r * np.cos(T)
            Y = r * np.sin(T)
            ax.plot_surface(X, Y, Z_grid, alpha=0.4, color='blue')

        elif ptype == 'shell':
            # Spherical shell (surface only)
            phi = np.linspace(0, np.pi, 20)
            theta = np.linspace(0, 2*np.pi, 30)
            P, T = np.meshgrid(phi, theta)
            r = 0.4
            X = r * np.sin(P) * np.cos(T)
            Y = r * np.sin(P) * np.sin(T)
            Z = r * np.cos(P)
            ax.plot_surface(X, Y, Z, alpha=0.4, color='blue')

        elif ptype == 'sphere':
            # Solid sphere (filled)
            phi = np.linspace(0, np.pi, 15)
            theta = np.linspace(0, 2*np.pi, 20)
            for r in [0.15, 0.3, 0.4]:
                P, T = np.meshgrid(phi, theta)
                X = r * np.sin(P) * np.cos(T)
                Y = r * np.sin(P) * np.sin(T)
                Z = r * np.cos(P)
                alpha = 0.1 if r < 0.4 else 0.3
                ax.plot_surface(X, Y, Z, alpha=alpha, color='blue')

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.6, 0.6)
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_primitives.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig4_primitives.png")


def fig5_tsr_chain():
    """Figure 5: TSR chain for articulated object (door with handle)."""
    fig = plt.figure(figsize=(12, 6))

    # Left: Conceptual diagram
    ax1 = fig.add_subplot(121)
    ax1.set_title('TSR Chain: Door Handle Grasp', fontsize=12, fontweight='bold')

    # Draw boxes representing frames
    boxes = [
        (0.1, 0.5, 'World\nFrame', '#E8E8E8'),
        (0.35, 0.5, 'Door\nFrame', '#FFE4B5'),
        (0.6, 0.5, 'Handle\nFrame', '#98FB98'),
        (0.85, 0.5, 'Grasp\nFrame', '#87CEEB'),
    ]

    for x, y, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x-0.08, y-0.12), 0.16, 0.24,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, edgecolor='black',
                                        linewidth=2)
        ax1.add_patch(rect)
        ax1.text(x, y, label, ha='center', va='center', fontsize=10,
                fontweight='bold')

    # Draw arrows between boxes
    arrows = [
        (0.18, 0.5, 0.27, 0.5, '$T_0^w$\n(door pose)'),
        (0.43, 0.5, 0.52, 0.5, '$T_w^e$\n(hinge→handle)'),
        (0.68, 0.5, 0.77, 0.5, '$T_w^e$\n(handle→grasp)'),
    ]

    for x1, y1, x2, y2, label in arrows:
        ax1.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=2))
        ax1.text((x1+x2)/2, y1+0.2, label, ha='center', va='bottom',
                fontsize=9, color='purple')

    # Add bounds annotations
    ax1.text(0.35, 0.25, '$B_w$: door angle\n[-π/2, 0]', ha='center',
             fontsize=9, style='italic', color='#666666')
    ax1.text(0.85, 0.25, '$B_w$: grasp\ntolerance', ha='center',
             fontsize=9, style='italic', color='#666666')

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')

    # Right: 3D visualization
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('3D View: Door at Different Angles', fontsize=12, fontweight='bold')

    # Draw door frame (fixed)
    door_hinge = np.array([0, 0, 0])
    draw_frame(ax2, door_hinge, np.eye(3), scale=0.2, labels=['', '', ''])

    # Draw door at different angles
    for angle, alpha in [(0, 0.3), (-np.pi/4, 0.5), (-np.pi/2, 0.8)]:
        R_door = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        # Door panel
        door_width = 0.6
        door_height = 0.8
        door_corners = np.array([
            [0, 0, 0],
            [door_width, 0, 0],
            [door_width, 0, door_height],
            [0, 0, door_height]
        ])
        rotated_corners = [door_hinge + R_door @ c for c in door_corners]
        door_face = Poly3DCollection([rotated_corners], alpha=alpha*0.3,
                                      facecolor='brown', edgecolor='black')
        ax2.add_collection3d(door_face)

        # Handle position
        handle_local = np.array([door_width - 0.1, 0, door_height * 0.5])
        handle_pos = door_hinge + R_door @ handle_local
        ax2.scatter(*handle_pos, s=50, c='gold', marker='o', alpha=alpha)

        # Draw small grasp frame at handle
        draw_frame(ax2, handle_pos, R_door, scale=0.1, labels=['', '', ''],
                   alpha=alpha, linewidth=1.5)

    ax2.set_xlim(-0.8, 0.8)
    ax2.set_ylim(-0.8, 0.8)
    ax2.set_zlim(0, 1.0)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=20, azim=-120)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_tsr_chain.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig5_tsr_chain.png")


def fig6_placement_example():
    """Figure 6: Practical example - mug placement on table."""
    fig = plt.figure(figsize=(12, 5))

    # Left: The template definition
    ax1 = fig.add_subplot(121)
    ax1.set_title('Template: Mug on Table', fontsize=12, fontweight='bold')

    template_text = """
position:
  type: plane
  x: [-0.15, 0.15]
  y: [-0.15, 0.15]
  z: 0

orientation:
  approach: -z    # mug upright
  yaw: free       # any rotation
    """

    ax1.text(0.05, 0.95, template_text, transform=ax1.transAxes,
             fontsize=11, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='gray'))

    # Add annotations
    ax1.annotate('Planar region\non table surface', xy=(0.55, 0.65),
                 xytext=(0.7, 0.75), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='blue'))
    ax1.annotate('Mug must be\nupright (-z approach)', xy=(0.55, 0.35),
                 xytext=(0.7, 0.25), fontsize=10,
                 arrowprops=dict(arrowstyle='->', color='green'))

    ax1.axis('off')

    # Right: 3D visualization
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Valid Placement Region', fontsize=12, fontweight='bold')

    # Draw table
    table_verts = [
        [(-0.4, -0.4, 0), (0.4, -0.4, 0), (0.4, 0.4, 0), (-0.4, 0.4, 0)]
    ]
    table = Poly3DCollection(table_verts, alpha=0.4, facecolor='#8B4513',
                             edgecolor='black', linewidth=2)
    ax2.add_collection3d(table)

    # Draw valid region (the plane constraint)
    x = np.linspace(-0.15, 0.15, 20)
    y = np.linspace(-0.15, 0.15, 20)
    X, Y = np.meshgrid(x, y)
    Z = np.ones_like(X) * 0.005
    ax2.plot_surface(X, Y, Z, alpha=0.5, color='#32CD32', edgecolor='none')

    # Draw constraint boundary
    ax2.plot([-0.15, 0.15, 0.15, -0.15, -0.15],
             [-0.15, -0.15, 0.15, 0.15, -0.15],
             [0.01, 0.01, 0.01, 0.01, 0.01], 'g-', linewidth=2)

    # Draw a few example mugs
    theta = np.linspace(0, 2*np.pi, 20)
    for pos, yaw in [((0, 0), 0), ((0.1, 0.08), np.pi/3), ((-0.08, 0.1), np.pi)]:
        z_cyl = np.linspace(0.01, 0.15, 10)
        theta_grid, z_grid = np.meshgrid(theta, z_cyl)
        r = 0.03
        x_cyl = pos[0] + r * np.cos(theta_grid + yaw)
        y_cyl = pos[1] + r * np.sin(theta_grid + yaw)
        ax2.plot_surface(x_cyl, y_cyl, z_grid, alpha=0.6, color='#4169E1')

    # Draw approach direction arrow
    ax2.quiver(0, -0.25, 0.25, 0, 0, -0.15, color='red', arrow_length_ratio=0.3,
               linewidth=2)
    ax2.text(0.05, -0.25, 0.3, 'approach: -z', color='red', fontsize=10)

    ax2.set_xlim(-0.4, 0.4)
    ax2.set_ylim(-0.4, 0.4)
    ax2.set_zlim(0, 0.4)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=25, azim=-60)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_placement_example.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("Generated: fig6_placement_example.png")


def main():
    """Generate all figures."""
    print("Generating TSR tutorial figures...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    fig1_tsr_concept()
    fig2_coordinate_frames()
    fig3_bounds_matrix()
    fig4_primitives()
    fig5_tsr_chain()
    fig6_placement_example()

    print()
    print("All figures generated successfully!")


if __name__ == '__main__':
    main()
