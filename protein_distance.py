import numpy as np
from Bio.PDB import PDBParser
import csv
from MDAnalysis import Universe

# Helper functions
def get_centroid(atom_coords):
    """Calculate the centroid of a set of coordinates."""
    return np.mean(atom_coords, axis=0)

def identify_membrane_leaflets(universe):
    """
    Identify membrane leaflets using phosphate positions, focusing on the lower membrane.
    Returns coordinates of headgroup atoms for both leaflets.
    """
    # Select phosphate and choline/serine/inositol atoms of all lipids
    headgroup_atoms = universe.select_atoms(
        "resname POPC POPS POPI and (name P PO4 NC3 NH3 C11 C12 C13 C14 C15)"
    )
    
    # Get z-coordinates and find the membrane centers
    z_coords = headgroup_atoms.positions[:, 2]
    
    # Use k-means to find two membrane centers
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2)
    Z = z_coords.reshape(-1, 1)
    kmeans.fit(Z)
    membrane_centers = sorted(kmeans.cluster_centers_.flatten())  # Sorted from bottom to top
    
    # Focus on lower membrane
    lower_membrane_center = membrane_centers[0]
    
    # Split into upper and lower leaflets of the lower membrane
    # Use a window around the lower membrane center
    window = 20  # Adjust this value based on membrane thickness
    lower_membrane_mask = (z_coords >= lower_membrane_center - window) & (z_coords <= lower_membrane_center + window)
    lower_membrane_atoms = headgroup_atoms.positions[lower_membrane_mask]
    
    lower_z_coords = lower_membrane_atoms[:, 2]
    lower_leaflet = lower_membrane_atoms[lower_z_coords < lower_membrane_center]
    upper_leaflet = lower_membrane_atoms[lower_z_coords > lower_membrane_center]
    
    return upper_leaflet, lower_leaflet

def fit_plane(points):
    """Fit a plane to a set of points using least squares."""
    if len(points) < 3:
        raise ValueError("Need at least 3 points to fit a plane")
        
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)  # Ax + By + D = Z
    normal_vector = np.array([-C[0], -C[1], 1])
    centroid = np.mean(points, axis=0)
    return centroid, normal_vector / np.linalg.norm(normal_vector)

def distance_point_to_plane(point, plane_centroid, plane_normal):
    """Calculate the perpendicular distance from a point to a plane."""
    return np.abs(np.dot(plane_normal, point - plane_centroid))

def save_plane_to_pml(centroid, normal, output_pml, protein_centroid=None, distance=None):
    """Generate a PyMOL script to visualize the membrane plane."""
    size = 50.0  # Adjust size as needed
    corners = [
        centroid + size * np.array([1, 1, 0]),
        centroid + size * np.array([-1, 1, 0]),
        centroid + size * np.array([-1, -1, 0]),
        centroid + size * np.array([1, -1, 0])
    ]

    with open(output_pml, 'w') as pml:
        pml.write('from pymol.cgo import *\n')
        pml.write('from pymol import cmd\n')
        # Write everything as a single command
        pml.write(f'obj = [BEGIN, TRIANGLE_FAN, COLOR, 1.0, 1.0, 0.0, NORMAL, {normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}, ')
        for corner in corners:
            pml.write(f'VERTEX, {corner[0]:.3f}, {corner[1]:.3f}, {corner[2]:.3f}, ')
        pml.write('END]; cmd.load_cgo(obj, "membrane_plane"); cmd.set("cgo_transparency", 0.5, "membrane_plane")')
        
        if protein_centroid is not None and distance is not None:
            pml.write(f'\n# Distance from protein centroid to membrane plane: {distance:.2f} Angstroms')

def main_dcd(topology_file, trajectory_file, output_csv, output_pml):
    """Process DCD trajectory and calculate membrane-protein distances."""
    u = Universe(topology_file, trajectory_file)
    distances = []

    # Process first frame to get the membrane plane
    protein = u.select_atoms('protein')
    protein_centroid = get_centroid(protein.positions)
    
    # Get membrane leaflets
    upper_leaflet, lower_leaflet = identify_membrane_leaflets(u)
    
    # Fit plane to lower leaflet
    membrane_centroid, membrane_normal = fit_plane(lower_leaflet)
    
    # Calculate distance
    distance = distance_point_to_plane(protein_centroid, membrane_centroid, membrane_normal)
    distances.append((0, distance))
    
    # Generate the PyMOL visualization including the distance
    save_plane_to_pml(membrane_centroid, membrane_normal, output_pml, 
                     protein_centroid=protein_centroid, distance=distance)
    
    print(f"\nResults for first frame:")
    print(f"Protein centroid coordinates: ({protein_centroid[0]:.2f}, {protein_centroid[1]:.2f}, {protein_centroid[2]:.2f})")
    print(f"Membrane plane centroid: ({membrane_centroid[0]:.2f}, {membrane_centroid[1]:.2f}, {membrane_centroid[2]:.2f})")
    print(f"Distance from protein centroid to membrane plane: {distance:.2f} Angstroms\n")

if __name__ == "__main__":
    topology_file = input("Enter the path to the topology (PSF/PDB) file: ")
    trajectory_file = input("Enter the path to the trajectory (DCD) file: ")
    output_csv = input("Enter the path to the output CSV file: ")
    output_pml = input("Enter the path to the output PML file: ")
    
    main_dcd(topology_file, trajectory_file, output_csv, output_pml)

#Load the membrane PDB file into pymol.
#then load:    
#from pymol.cgo import *
#from pymol import cmd
#then, open @membrane_plane.pml







