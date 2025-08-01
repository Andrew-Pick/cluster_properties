import numpy as np

class SearchGrid:
    '''
    Class to enable fast finding of particles within a certain radius around a position in a hdf5 snapshot.
    The module bins the simulation partciles onto a grid (which takes extra time) and then uses this grid 
    to narrow down the search area around a specific position which leads to a significant speedup for large 
    simulations. 
    '''
    def __init__(self, 
                 snapshot, 
                 ptypes:list = [0,1,2,3,4,5], 
                 gridsize:int = 8, 
                 coordinate_name:str = 'Coordinates'):
        '''
        Initialise the search grid object and bin the particles onto a grid. 

        Args:
            snapshot: a snapshot object from which to find particles. 
            parttypes: the partcile types to consider, requires list of int
            gridsize: the gridsize to use for the search grid, smaller grids 
            are faster to generate but don't lead to the same speedup as large 
            grids during the search
            coordinate_name: the name of the Coordinates in the snapshot, either 
            'Coordinates' or 'IntegerCoordinates'
        '''
        self.s = snapshot
        self.gridsize = gridsize
        self.coordinate_name = coordinate_name
        self.spacing = self.s.header.boxsize / self.s.const.h / self.gridsize
        self.grid = {}

        for pt in ptypes:
            self.bin_particles(self.s.get_part_type_name(pt))

        
    def bin_particles(self, parttype_name):
        print("Binning particles.")
        self.create_grid(parttype_name)
        self.calculate_cell_indices(parttype_name)

        for i in np.arange(self.gridsize):
            print("Processing column ", i)
            ipart = np.where(self.index_3d[:, 0] == i)[0]
            for j in np.arange(self.gridsize):
                jipart = np.where(self.index_3d[ipart, 1] == j)[0]
                for k in np.arange(self.gridsize):
                    part = np.where(self.index_3d[ipart[jipart], 2] == k)[0]
                    self.grid[parttype_name][i, j, k].add_particles(ipart[jipart[part]])


    def calculate_cell_indices(self, parttype_name):
        print("Calculating cell indices.")
        self.index_3d = (self.s.data[self.coordinate_name][parttype_name] / self.spacing).astype(int)
        self.index_1d = self.index_3d[:,0] * self.gridsize**2 + self.index_3d[:,1] * self.gridsize + self.index_3d[:,2]


    def create_grid(self, parttype_name):
        print("Creating grid for ", parttype_name)
        self.grid[parttype_name] = np.zeros((self.gridsize, self.gridsize, self.gridsize), dtype = object)
        self.grid_left = np.zeros((self.gridsize, self.gridsize, self.gridsize, 3))
        self.grid_right = np.zeros((self.gridsize, self.gridsize, self.gridsize, 3))

        for i in np.arange(self.gridsize):
            for j in np.arange(self.gridsize):
                for k in np.arange(self.gridsize):
                    self.grid[parttype_name][i, j, k] = GridCell(self, np.array([i, j, k]))
                    self.grid_left[i, j, k] = np.array([i, j, k]) * self.spacing
                    self.grid_right[i, j, k] = (np.array([i, j, k])+1) * self.spacing
                    

    def find_cells_no_boxwrap(self, center, radius):
        left = center - radius
        right = center + radius

        cells = np.where((left[0] < self.grid_right[:,:,:,0]) & (right[0] > self.grid_left[:,:,:,0]) & (left[1] < self.grid_right[:,:,:,1]) & (right[1] > self.grid_left[:,:,:,1]) & (left[2] < self.grid_right[:,:,:,2]) & (right[2] > self.grid_left[:,:,:,2]))    
        cells = np.array(cells).swapaxes(0,1)
        return cells

    def find_cells(self, center, radius, parttype):
        box = self.s.header.boxsize / self.s.const.h
        left = center - radius
        right = center + radius

        cells = self.find_cells_no_boxwrap(center, radius)

        #extra cells for periodic boundaries
        if any(left < 0) or any(right > box):
            for oi in [-box, 0, box]:
                for oj in [-box, 0, box]:
                    for ok in [-box, 0, box]:
                        offset = np.array([oi, oj, ok])
                        
                        if all(offset == 0):
                            continue

                        cells_bw = self.find_cells_no_boxwrap(center + offset, radius)
                        cells = np.concatenate((cells, cells_bw))

        return cells

    def find_particles(self, center, radius, parttype):
        cells = self.find_cells(center, radius, parttype)
        box = self.s.header.boxsize / self.s.const.h

        part = np.array([], dtype = np.int64)
        for cell in cells:
            rad = (center - self.s.data[self.coordinate_name][parttype][self.grid[parttype][cell[0], cell[1], cell[2]].cell_particles])
            rad[rad < -box/2.] += box
            rad[rad >= box/2.] -= box
            radius_sq = np.sum((rad)**2, axis = 1)
            ind = np.where(radius_sq <= radius**2)[0]
            new_part = self.grid[parttype][cell[0], cell[1], cell[2]].cell_particles[ind]
            part = np.concatenate((part, new_part))

        return np.sort(part)



class GridCell:
    def __init__(self, search_grid, index_3d):
        self.sg = search_grid
        self.cell_index_3d = index_3d
        self.cell_index_1d = index_3d[0] * self.sg.gridsize**2 + index_3d[1] * self.sg.gridsize + index_3d[2]
        self.cell_particles = np.array([], dtype = np.int64)        

    def add_particles(self, particles):
        self.cell_particles = np.concatenate((self.cell_particles, particles))






if __name__ == "__main__":
    import timeit
    import imp
    from read_hdf5 import snapshot

    s = snapshot(26, 'test_snapshot/', verbose = True)
    s.read(['Coordinates'])
    print('setting up search grid')
    t0 = timeit.default_timer()
    sg = SearchGrid(s, ptypes = [1], gridsize = 8, coordinate_name = 'Coordinates')
    t1 = timeit.default_timer()
    print('took', t1 - t0, 's')

    tot_new = 0
    tot_old = 0
    for t in range(100):    
        center = np.random.random(3) * s.header.boxsize/s.const.h
        radius = np.random.random(1)[0]* s.header.boxsize/s.const.h / 50

        print("center", center, "radius", radius)

        #print("old method")
        start = timeit.default_timer()

        box = s.header.boxsize / s.const.h
        r = center - s.data['Coordinates']['dm']
        r[r < -box/2.] += box
        r[r > box/2.] -= box
        r_sq = np.sum((r)**2, axis = 1)
        old_part = np.where(r_sq < radius**2)[0]

        stop = timeit.default_timer()
        #print("took", stop - start, "s")
        tot_old += stop - start

        #print("new method")
        start = timeit.default_timer()

        new_part = sg.find_particles(center, radius, 'dm')

        stop = timeit.default_timer()
        #print("took", stop - start, "s")

        e = 1
        if len(old_part)==len(new_part):
            if len(old_part > 0):
                if np.unique(old_part==new_part)[0]:
                    #print("done", t)
                    e = 0
            else:
                e=0

        if e == 1:
            print("---------------------------")
            print(old_part)
            print(new_part)
            print("error", center, radius)
            print("---------------------------")
            break


