import main



#
# MODEL-SETUP
#
kg_model=main.get_kg_model()
kg_model.load_weights(main.BASE_WEIGHTS)
kg_model.graph
kg_model.model().summary()



#
# MODEL-RUN
#
RUN_NAME='full-test'
kg_model.fit_gen(
     epochs=1,
     train_gen=main.train_gen,
     train_steps=20,
     validation_gen=main.valid_gen,
     validation_steps=10,
     history_name=RUN_NAME,
     checkpoint_name=RUN_NAME)

