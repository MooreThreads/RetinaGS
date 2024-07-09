'''
some comments says that scripts like train_MP_tree_partition.py are not easy to read
thus burning decided to format some trainer which would make distributed training more undersatndable:
    + hide details to make the code explain itself
    + reuse some logic to make code more concise

but exchanging GS model would be left as a mess, by the way, a elegant implementation shall:
    + utilize disk, RAM to avoid OOM
    + proper async methods  
'''