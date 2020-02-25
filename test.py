import sys
fname = 'job_{}_arrayid_{}_Ca_{}'.format(sys.argv[1], sys.argv[2])
print('arg 1: {}, arg 2: {}'.format(sys.argv[1], sys.argv[2]))

# with open(fname, 'wb') as f:
#     pickle.dump({'mesh_name': meshfile, 'data': res, 'bin_middists': middists}, f)