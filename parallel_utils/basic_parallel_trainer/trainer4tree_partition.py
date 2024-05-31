'''
some comments says that scripts like train_MP_tree_partition.py are not easy to read
thus burning decided to format some trainer which would make distributed training more undersatndable
but burning insist that exchanging GS model would be left as a mess:
    + the poovided implementaton use the same control logic as exchanging other msg with small size
    + exchange model can easily cause OOM!       
    + design your own mechanism for production usage
'''


class Trainer4TreePartition():
    def __init__(self, ) -> None:
        pass