# ESA_Ariel_Hackathon_2025

I was able to attend the ESA Datalabs Ariel Hackathon in ESAC (European Space Astronomy Centre) Madrid.

By using simulated data for the future Ariel Mission (set to be launched in 2029), we developed machine learning algorithms to improve the perfomance while predicting the relative radii of different exoplanets with various degrees of stellar spots and gaussian photon noise. 

For two days, along with Rupesh Durgesh and Rebekka Koch, I developed various different machine learning algorithms with the aim of improving the provided Ridge algorithms' score on a test set (9058).

Initially, I decided to produce a Lasso algorithm with the aim of comparing the perfomance of Lasso (L1 regularisation) and the provided Ridge (L2 regularisation). The Lasso algorithm was observed to perfom slightly worst than Ridge with a score of 9051.

Therefore, the next sensible algorithm to attempt a higher score was the Bayesian Ridge, using MultiOutputRegressor(BayesianRidge(...) for applying Bayesian Ridge to each of the 55 target variables (relative radii estimated at each of the 55 different wavelength channels). It led to obtaining a score of 9166, performing signficantly better than the original Ridge.

Finally, an attempt was made to produce a Neural Network using PyTorch. Using two hidden layers with relu activation functions containing 1024 and 256 units, respectively, and an output of 55 units. ADAM was used as the optimizer. The score obtained from this NN was 9094 which was higher than for both the Ridge and Lasso but lower than that of the Bayesian Ridge. I believe that with some further fine-tuning of the parameters this score could have been significantly improved.
