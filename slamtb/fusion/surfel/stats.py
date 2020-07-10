class FusionStats:
    """Surfel fusion step statistics.

    Attributes:

       added_count (int): How many surfels were added in the step.

       merged_count (int): How many surfels were merged in the step.

       removed_count (int): How many surfels were removed in the step.
    """

    def __init__(self, added_count=0, merged_count=0, removed_count=0, carved_count=0):
        self.added_count = added_count
        self.merged_count = merged_count
        self.removed_count = removed_count
        self.carved_count = carved_count
        
    def __str__(self):
        return "Fusion stats: {} added, {} merged, {} removed, {} carved".format(
            self.added_count, self.merged_count, self.removed_count, self.carved_count)

    def __repr__(self):
        return str(self)
