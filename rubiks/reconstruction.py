class CubeReconstructor:
    def __init__(self, input_faces):
        self.input_faces = input_faces  # List of tuples: (face_label, face_string)
        self.faces = {}  # Placed face strings
        self.used_corners = set()
        self.used_edges = set()
        self.precomputed_rotations = self._precompute_rotations()
        
        self.valid_corner = {"WRB", "WRG", "WGO", "WBO", "YRB", "YRG", "YGO", "YBO"}
        self.valid_edge = {"WR", "WG", "WB", "WO", "RB", "RG", "OB", "OG", "YR", "YG", "YB", "YO"}

        self.corner_mappings = {
            "WRB": (6, 2, 0),
            "WRG": (0, 0, 2),
            "WGO": (2, 0, 2),
            "WBO": (8, 2, 0),
            "YRB": (0, 8, 6),
            "YRG": (6, 6, 8),
            "YGO": (8, 6, 8),
            "YBO": (2, 8, 6)
        }

        self.edge_mappings = {
            "WR": (3, 1),
            "WG": (1, 1),
            "WB": (7, 1),
            "WO": (5, 1),
            "RB": (5, 3),
            "RG": (3, 5),
            "OB": (3, 5),
            "OG": (5, 3),
            "YR": (3, 7), 
            "YG": (7, 7),
            "YB": (1, 7),
            "YO": (5, 7)
        }

    def _precompute_rotations(self):
        rotations = {}
        for label, face_str in self.input_faces:
            face_rotations = [face_str]
            current = list(face_str)
            for _ in range(3):
                current = [
                    current[6], current[3], current[0],
                    current[7], current[4], current[1],
                    current[8], current[5], current[2]
                ]
                face_rotations.append(''.join(current))
            rotations[label] = face_rotations
        return rotations

    def reconstruct(self):
        stack = self.input_faces.copy()
        return self._backtrack(stack, [])

    def _backtrack(self, stack, placed_faces):
        if not stack:
            return True  # Success

        face_label, _ = stack.pop()

        for rotated_face in self.precomputed_rotations[face_label]:
            if self._place_face(face_label, rotated_face):
                placed_faces.append((face_label, rotated_face))

                if self._backtrack(stack, placed_faces):
                    return True

                self._remove_face(face_label)
                placed_faces.pop()

        stack.append((face_label, _))  # Restore for higher-level retry
        return False

    def _place_face(self, label, face_str):
        if label in self.faces:
            return False  # Already placed

        self.faces[label] = face_str
        if not self._is_valid(label):
            del self.faces[label]
            return False

        return True

    def _remove_face(self, label):
        del self.faces[label]
        self.used_corners.clear()
        self.used_edges.clear()
        for face_label, face_str in self.faces.items():
            self._update_used_cubies(face_label, face_str)

    def _update_used_cubies(self, label, face_str):
        for combo, (i1, i2, i3) in self.corner_mappings.items():
            if label in combo and all(f in self.faces for f in combo):
                colors = [self.faces[f][i] for f, i in zip(combo, (i1, i2, i3))]
                self.used_corners.add(''.join(sorted(colors)))

        for combo, (i1, i2) in self.edge_mappings.items():
            if label in combo and all(f in self.faces for f in combo):
                colors = [self.faces[f][i] for f, i in zip(combo, (i1, i2))]
                self.used_edges.add(''.join(sorted(colors)))

    def _is_valid(self, label):
        local_corners = set()
        local_edges = set()

        for combo, (i1, i2, i3) in self.corner_mappings.items():
            if label in combo:
                try:
                    colors = [self.faces[f][i] for f, i in zip(combo, (i1, i2, i3))]
                    cubie = ''.join(sorted(colors))
                    if len(set(colors)) != 3 or cubie in self.used_corners or cubie not in self.valid_corner:
                        return False
                    local_corners.add(cubie)
                except:
                    continue

        for combo, (i1, i2) in self.edge_mappings.items():
            if label in combo:
                try:
                    colors = [self.faces[f][i] for f, i in zip(combo, (i1, i2))]
                    cubie = ''.join(sorted(colors))
                    if len(set(colors)) != 2 or cubie in self.used_edges or cubie not in self.valid_edge:
                        return False
                    local_edges.add(cubie)
                except:
                    continue

        self.used_corners.update(local_corners)
        self.used_edges.update(local_edges)
        return True

    def get_state(self):
        return self.faces
