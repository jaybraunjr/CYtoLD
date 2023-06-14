import MDAnalysis as mda 
import pandas as pd
from MDAnalysis.analysis import distances
from MDAnalysis.analysis import contacts
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde as kde
import matplotlib.pyplot as plt
from MDAnalysis.coordinates import XTC


class contact_analysis:
    def __init__(self, universe, residue_ids, pl, nl, radius=3.5):
        self.universe = universe
        self.residue_ids = residue_ids
        self.pl = pl
        self.nl = nl
        self.radius = radius

    def contacts_within_cutoff(self, group_a, group_b):
        dist = distances.distance_array(group_a.positions, group_b.positions)
        contact_mtx = contacts.contact_matrix(dist, self.radius)
        return contact_mtx.sum()

    def analyze_contacts(self):
        result = []
        n_residues = len(self.residue_ids)
        n_frames = len(self.universe.trajectory)
        
        pl_contacts = np.zeros((n_residues, n_frames))
        nl_contacts = np.zeros((n_residues, n_frames))

        residues = [self.universe.select_atoms(f"(resid {resid}) and (not backbone)") for resid in self.residue_ids]

        for ts_idx, ts in enumerate(self.universe.trajectory):
            for resid_idx, residue in enumerate(residues):
                pl_contacts[resid_idx, ts_idx] = self.contacts_within_cutoff(residue, self.pl)
                nl_contacts[resid_idx, ts_idx] = self.contacts_within_cutoff(residue, self.nl)

        result = np.vstack((pl_contacts, nl_contacts))
        return result



class depth_analysis:
    def __init__(self, traj, protein_ca_indices, lipid_p_indices):
        self.traj = traj
        self.protein_ca_indices = protein_ca_indices
        self.lipid_p_indices = lipid_p_indices

    def compute_penetration_depth(self):
        lipid_p_positions = self.traj.xyz[:, self.lipid_p_indices]

        lipid_p_indices_top = self.lipid_p_indices[lipid_p_positions[..., 2].mean(axis=0) >= self.traj.unitcell_lengths[:, 2].mean() / 2]
        lipid_p_indices_bottom = self.lipid_p_indices[lipid_p_positions[..., 2].mean(axis=0) < self.traj.unitcell_lengths[:, 2].mean() / 2]

        lipid_p_com_top = md.compute_center_of_mass(self.traj.atom_slice(lipid_p_indices_top))
        lipid_p_com_bottom = md.compute_center_of_mass(self.traj.atom_slice(lipid_p_indices_bottom))

        protein_ca_positions = self.traj.xyz[:, self.protein_ca_indices]

        penetration_depth = np.empty((self.traj.n_frames, len(self.protein_ca_indices)))

        for frame in range(self.traj.n_frames):
            dist_top = np.linalg.norm(protein_ca_positions[frame] - lipid_p_com_top[frame], axis=-1)
            dist_bottom = np.linalg.norm(protein_ca_positions[frame] - lipid_p_com_bottom[frame], axis=-1)
            binding_to_top = dist_top < dist_bottom

            penetration_depth[frame] = np.where(binding_to_top, protein_ca_positions[frame, :, 2] - lipid_p_com_top[frame, 2],
                                                protein_ca_positions[frame, :, 2] - lipid_p_com_bottom[frame, 2])

            box_half_z = self.traj.unitcell_lengths[frame, 2] / 2

            for i, depth in enumerate(penetration_depth[frame]):
                if depth > 0 and protein_ca_positions[frame, i, 2] < lipid_p_com_bottom[frame, 2] and not binding_to_top[i]:
                    penetration_depth[frame, i] = -depth
                elif depth < 0 and protein_ca_positions[frame, i, 2] < box_half_z:
                    penetration_depth[frame, i] = np.abs(depth)

                if depth > 0 and protein_ca_positions[frame, i, 2] < box_half_z and not binding_to_top[i]:
                    penetration_depth[frame, i] = -depth

        return penetration_depth




class DepthAnalysis:

    # Here we are analyzing the depth of the protein in the membrane

    def __init__(self, gro_file, xtc_file):
        self.u = mda.Universe(gro_file, xtc_file)
        self.halfz = self.u.dimensions[2] / 2
        self.UP = self.u.select_atoms('name P and prop z > %f' % self.halfz)
        self.LP = self.u.select_atoms('name P and prop z < %f' % self.halfz)

    def initial(self, prot):
        text = []
        for ts in self.u.trajectory:
            t = self.u.trajectory.time / 1000
            proteinZ = prot.center_of_mass()[2]
            trilayerLP = self.LP.center_of_mass()[2]
            trilayerUP = self.UP.center_of_mass()[2]
            proteinCoord = proteinZ - trilayerUP
            if proteinCoord > -20:
                text.append([t, proteinCoord])
            else:
                proteinCoord = trilayerLP - proteinZ
                text.append([t, proteinCoord])
        arr = np.array(text)
        return arr


    def run_analysis(self, start_resid, end_resid, exclude_resid_list=None, output_file='output_all_residues.csv'):
        all_residues_data = []

        for resid in range(start_resid, end_resid + 1):
            if exclude_resid_list and resid in exclude_resid_list:
                continue
            protein = self.u.select_atoms(f'not backbone and not name H* and resid {resid}')
            resname = protein.residues[0].resname
            go = self.initial(protein)

            # In this loop, we are appending the time and z-coordinate of each residue to the all_residues_data list
            for idx, (time, z_coord) in enumerate(go):
                # In this if statement, we are checking if the length of the all_residues_data list is less than the index of the current residue
                if len(all_residues_data) <= idx:
                    # If the length of the all_residues_data list is less than the index of the current residue, we append a new list to the all_residues_data list
                    all_residues_data.append([time])
                all_residues_data[idx].append(z_coord)

        columns = ['time(ps)'] + [f'{resid} {resname}' for resid, resname in zip(range(start_resid, end_resid + 1), [res.resname for res in self.u.select_atoms('protein').residues[start_resid-1:end_resid]]) if not (exclude_resid_list and resid in exclude_resid_list)]
        df = pd.DataFrame(all_residues_data, columns=columns)
        df.to_csv(output_file, index=False)



class MembraneAnalyzer:
    def __init__(self, gro_file, dcd_file):
        self.u = mda.Universe(gro_file, dcd_file)
        self.protein = self.u.select_atoms('protein')

    def calculate_center_of_mass(self):
        return np.mean(self.protein.positions, axis=0)

    def select_phospholipid_atoms(self, halfz):
        pl = self.u.select_atoms('resname POPC DOPE SAPI and name P and around 10 protein and prop z < %f' % halfz, updating=True)
        print(len(pl))
        return pl

    def calculate_average_z(self, LP):
        return np.mean(LP.positions[:, 2])

    def identify_defect_atoms(self, defect_threshold):
        mask = self.protein.positions[:, 2] > defect_threshold
        return self.protein[mask]

    def calculate_defect_area_volume(self, defect_protein):
        defect_vol = ConvexHull(defect_protein.positions).volume
        defect_area = ConvexHull(defect_protein.positions[:, :2]).area
        return defect_area, defect_vol

    def plot_hull(self, defect_protein_xy, ts):
        hull = ConvexHull(defect_protein_xy)
        plt.figure()
        for simplex in hull.simplices:
            plt.plot(defect_protein_xy[simplex, 0], defect_protein_xy[simplex, 1], 'k-')
        plt.title(f'Convex Hull at frame {ts.frame}')
        plt.show()

    def generate_heatmap(self, all_points):
        k = kde([all_points[:, 0], all_points[:, 1]])
        xi, yi = np.mgrid[0:self.u.dimensions[0]:1000j, 0:self.u.dimensions[1]:1000j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        threshold = 0.000001
        zi_adj = np.where(zi > threshold, zi, np.nan)
        plt.figure(figsize=(8, 8))
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', cmap='jet', vmax=0.003, vmin=0.0003)
        plt.colorbar(label='Density')
        plt.xlim(0, self.u.dimensions[0])
        plt.ylim(0, self.u.dimensions[1])
        plt.xlabel('x (Å)')
        plt.ylabel('y (Å)')
        plt.title('Packing defect density heat map')
        plt.savefig('packing_density_DCD.png', dpi=400)
        plt.show()

    def calculate_average_depths(self, depth_sums, depth_counts):
        return np.where(depth_counts != 0, depth_sums / depth_counts, np.nan)

    def generate_depth_heatmap(self, average_depths):
        plt.figure(figsize=(8, 8))
        plt.imshow(average_depths.T, origin='lower', extent=(0, self.u.dimensions[0], 0, self.u.dimensions[1]), cmap='hot', vmin=-8, vmax=0)
        plt.colorbar(label='Average Depth (Å)')
        plt.xlabel('x (Å)')
        plt.ylabel('y (Å)')
        plt.title('Average Depth Heat Map')
        plt.savefig('depth_mlx.png', dpi=400)
        plt.show()

    def main(self):
        ls = []
        all_points = []
        depth_sums = np.zeros((100, 100))
        depth_counts = np.zeros((100, 100), dtype=int)

        for ts in self.u.trajectory:
            halfz = self.u.dimensions[2] / 2
            LP = self.select_phospholipid_atoms(halfz)
            average_z = self.calculate_average_z(LP)
            defect_threshold = average_z 
            defect_protein = self.identify_defect_atoms(defect_threshold)
            defect_protein_xy = defect_protein.positions[:, :2]
            if defect_protein.n_atoms >= 10:
                defect_area, defect_vol = self.calculate_defect_area_volume(defect_protein)
                ls.append(defect_area)
                hull = ConvexHull(defect_protein_xy)
                all_points.extend(defect_protein_xy[simplex] for simplex in hull.simplices)

                for i in range(defect_protein.n_atoms):
                    x_bin = int(defect_protein.positions[i, 0] / self.u.dimensions[0] * 100)
                    y_bin = int(defect_protein.positions[i, 1] / self.u.dimensions[1] * 100)
                    depth =  defect_protein.positions[i, 2] - average_z
                    depth_sums[x_bin, y_bin] += depth
                    depth_counts[x_bin, y_bin] += 1

        all_points = np.concatenate(all_points)
        self.generate_heatmap(all_points)
        average_depths = self.calculate_average_depths(depth_sums, depth_counts)
        self.generate_depth_heatmap(average_depths)
        return ls


class CenterTraj:
    def init(self, gro_file, xtc_file, output):
        self.u = mda.Universe(gro_file, xtc_file)
        self.protein = self.u.select_atoms('protein')
        self.center = self.u.dimensions[:2] / 2.0
        self.output = output

        def translate_protein(self):
            with XTC.XTCWriter(self.output_file, self.u.atoms.n_atoms) as W:
                for ts in u.trajectory:
                    # Calculate the center of mass of the protein in the x-y plane
                    com = protein.center_of_mass()[:2]
                    # Calculate the translation vector, leaving z unchanged
                    translation_vector = center - com
                    translation_vector = [*translation_vector, 0.0]

                    # Translate the protein so its center of mass is at the center of the box in the x-y plane
                    protein.translate(translation_vector)
                    
                    # Write the entire Universe (including all molecules) to the trajectory
                    W.write(self.u.atoms)


