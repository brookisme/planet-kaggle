import main



#
# MODEL-SETUP
#
kg_model=main.get_kg_model()
kg_model.load_weights(main.BASE_WEIGHTS)
kg_model.model().summary()
kg_model.save('bn_res_brvw_17.hdf5')

