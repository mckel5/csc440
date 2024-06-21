import imagematrix

# a pixel, represented in column-major order
Pixel = tuple[int, int]
# a list of coordinates, e.g. `[(5, 0), (5, 1), (4, 2), (5, 3), (6, 4)]`
Path = list[Pixel]
# a total cost paired with a specific path
Seam = tuple[int, Path]


class ResizeableImage(imagematrix.ImageMatrix):
    _energy_cache: list[list[int | None]]
    _seams: list[Seam]

    def best_seam(self, dp=True) -> Path:
        """
        Calculate the "best seam" of an image -- that is, find the line of
        pixels that would cause the smallest visual disruption of the image
        if removed.
        """

        # Memoized cache of energy values
        self._energy_cache = [
            [None for _ in range(self.height)] for _ in range(self.width)
        ]

        if dp:
            return self._best_seam_dynamic()
        else:
            # All seams will be stored here
            self._seams = []

            # Start the algorithm from each pixel in the first row, left to right
            for i in range(self.width):
                pixel = (i, 0)
                self._best_seam_recursive(pixel, (0, [pixel]))
            # Sort the seams by total cost, in descending order
            self._seams.sort(key=lambda seam: seam[0], reverse=True)
            # Get the lowest energy seam and return its path
            return self._seams.pop()[1]

    def remove_best_seam(self):
        self.remove_seam(self.best_seam())

    def _best_seam_recursive(self, pixel, seam):
        """
        Recursive algorithm for calculating the best seam.
        Consumes extreme amounts of memory when supplied with anything other than a tiny image.
        """

        cost_so_far, path = seam
        col, row = pixel

        # Base case: bottom of image reached
        if row == self.height - 1:
            self._seams.append(seam)
            return

        # Continue algorithm down and left, straight down, and down and right
        for col_offset in (-1, 0, 1):
            next_pixel = (col + col_offset, row + 1)

            # Make sure desired pixel is in bounds
            if not self._in_bounds(next_pixel):
                continue

            # Call recursively on next pixel
            next_cost = self._get_energy(next_pixel)
            new_path = path.copy()
            new_path.append(next_pixel)
            self._best_seam_recursive(next_pixel, (cost_so_far + next_cost, new_path))

    def _best_seam_dynamic(self):
        """
        Dynamic programming algorithm for calculating the best seam.
        """

        # Initialize DP table
        dp = [[None for _ in range(self.height)] for _ in range(self.width)]

        # Fill first row of DP table
        for i in range(self.width):
            dp[i][0] = (self._get_energy((i, 0)), [(i, 0)])

        # Fill DP table for each pixel in the image
        for row in range(1, self.height):
            for col in range(self.width):
                seam_choices = []

                # Check upper-left, upper, and upper-right pixels to find
                # the least costly seam so far
                for col_offset in (-1, 0, 1):
                    previous_col = col + col_offset
                    previous_row = row - 1

                    # Make sure desired pixel is in bounds
                    if not self._in_bounds((previous_col, previous_row)):
                        continue

                    seam_choices.append(dp[previous_col][previous_row])

                # Choose least costly seam from the upper pixels
                old_cost, old_path = min(seam_choices)

                # Update table at current row & col
                dp[col][row] = (
                    old_cost + self._get_energy((col, row)),
                    old_path.copy() + [(col, row)],
                )

        # Return least costly seam from bottom row
        bottom_row = [dp[i][self.height - 1] for i in range(self.width)]
        _, best_path = min(bottom_row)
        return best_path

    def _get_energy(self, pixel: Pixel) -> int:
        """
        Return the energy value of a pixel and memoize the result, if not already memoized.
        """

        col, row = pixel

        if self._energy_cache[col][row] is None:
            self._energy_cache[col][row] = self.energy(col, row)

        return self._energy_cache[col][row]

    def _in_bounds(self, pixel: Pixel) -> bool:
        col, row = pixel
        return (0 <= col < self.width) and (0 <= row < self.height)
