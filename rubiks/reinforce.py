class CubieEnforcer:
    def __init__(self, cube_reconstructor, cubie_constraints):
        """
        cube_reconstructor: an instance of CubeReconstructor (already solved state)
        cubie_constraints: list of tuples like ("WRB", "WRB") where first is cubie name, second is location
        """
        self.cube = cube_reconstructor
        self.constraints = cubie_constraints
        self.faces = cube_reconstructor.faces.copy()
        self.face_rotations = cube_reconstructor.precomputed_rotations
        self.corner_mappings = cube_reconstructor.corner_mappings
        self.edge_mappings = cube_reconstructor.edge_mappings
        self.valid_corner = cube_reconstructor.valid_corner
        self.valid_edge = cube_reconstructor.valid_edge

        self.fixed_cubies = []  # Stack of successfully placed cubies
        self.rotation_cache = {}  # Cache tested combinations for each cubie

    def enforce_constraints(self):
        return self._enforce_recursive(0)

    def _enforce_recursive(self, idx):
        if idx >= len(self.constraints):
            return True

        cubie, location = self.constraints[idx]
        is_corner = len(cubie) == 3
        mapping = self.corner_mappings if is_corner else self.edge_mappings

        if cubie not in mapping:
            return False

        face_ids = list(cubie)
        indices = mapping[cubie]

        # Cache key for this cubie
        cache_key = (cubie, location)
        if cache_key not in self.rotation_cache:
            self.rotation_cache[cache_key] = self._rotation_product(face_ids)

        original_faces = {f: self.faces[f] for f in face_ids}

        for rotations in self.rotation_cache[cache_key]:
            for f, r in zip(face_ids, rotations):
                self.faces[f] = self.face_rotations[f][r]

            correct = all(self.faces[face_ids[i]][indices[i]] == cubie[i] for i in range(len(cubie)))
            if not correct:
                continue

            if not self._is_still_valid():
                continue

            self.fixed_cubies.append((cubie, location, rotations))
            if self._enforce_recursive(idx + 1):
                return True
            self.fixed_cubies.pop()

        # Backtrack
        for f in face_ids:
            self.faces[f] = original_faces[f]

        return False

    def _rotation_product(self, face_ids):
        from itertools import product
        return list(product(range(4), repeat=len(face_ids)))

    def _is_still_valid(self):
        seen_corners = set()
        seen_edges = set()

        for cubie, indices in self.corner_mappings.items():
            try:
                colors = [self.faces[f][i] for f, i in zip(cubie, indices)]
                name = ''.join(sorted(colors))
                if name in seen_corners or name not in self.valid_corner:
                    return False
                seen_corners.add(name)
            except:
                continue

        for cubie, indices in self.edge_mappings.items():
            try:
                colors = [self.faces[f][i] for f, i in zip(cubie, indices)]
                name = ''.join(sorted(colors))
                if name in seen_edges or name not in self.valid_edge:
                    return False
                seen_edges.add(name)
            except:
                continue

        return True

    def get_final_state(self):
        return self.faces if self.fixed_cubies else None
