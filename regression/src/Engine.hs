{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE FlexibleInstances #-}
module Engine where

import Control.Monad
import Control.Monad.Random
import Data.Bifunctor
import Data.Functor.Identity
import Data.List.Split
import Data.Foldable
import Data.Maybe
import Data.Proxy
import qualified Data.Vector as V
import Linear
import Linear.V
import Model
import Models
import Optimizer
import Cost
import System.Random.Shuffle

-- | A linear (affine) model mapping from an n-dimensional space to scalars
type LinearModel n = AffineMap (V n) Identity

-- | A logistic model mapping from an n-dimensional space to scalars.
type LogisticModel n = AffineMap (V n) Identity :-> Over Identity "logistic"

type RawInput = [Double]
type RawOutput = Double

data OptimizerChoice = GradientDescent | AdaGrad | RMSProp

data Settings = Settings
    { trainData :: [(RawInput, RawOutput)]
    , testData :: [RawInput]
    , learningRate :: Double
    , learningRateDecay :: Double
    , shuffleTrain :: Bool
    , holdoutSize :: Int
    , batchSize :: Maybe Int
    , numIter :: Int
    , useLogistic :: Bool
    , l2Coef :: Double
    , standardizeInput :: Bool
    , optimizerChoice :: OptimizerChoice
    }

-- | The first phase of dispatch: reads the data dimensions and calls 'run'' with the correct
-- type.
run :: MonadRandom m => Settings -> m (Double, [RawOutput])
run settings = reifyDim (length $ fst $ head $ trainData settings) $ run' settings

-- | The second phase of dispatch: it gets the dimension passed quasi-dependently, on type-level,
-- formats the train and test data accordingly and dispatches 'run''' with the correct model and 
-- cost function (linear or logistic).
run' :: forall proxy n m. (MonadRandom m, Dim n) => Settings -> proxy n -> m (Double, [RawOutput])
run' settings _
    | useLogistic settings = second (runIdentity <$>) <$>
        run'' (zero :: LogisticModel n Double) (mean logistic + realToFrac (l2Coef settings) * l2)
              settings testDataV trainDataV

    | otherwise            = second (runIdentity <$>) <$>
        run'' (zero :: LinearModel n Double)   (mean square   + realToFrac (l2Coef settings) * l2)
              settings testDataV trainDataV
  where
    testDataV :: [V n Double]
    testDataV = fromJust . fromVector . V.fromList <$> testData settings

    trainDataV :: [(V n Double, Identity Double)]
    trainDataV = bimap (fromJust . fromVector . V.fromList) Identity <$> trainData settings

-- | The third phase of the dispatch: it gets the model and cost function and performs
-- the required computations.
run'' :: (Model m, Additive (Input m), Traversable m, Additive m, MonadRandom rm)
        => m Double -> Cost m -> Settings -> [Input m Double]
        -> [(Input m Double, Output m Double)] -> rm (Double, [Output m Double])
run'' model0 cost Settings{..} test train = do
    let (test', train') = (if standardizeInput then standardize else (,)) test train
    (holdout, train'') <- splitAt holdoutSize <$> shuffleM train'

    -- | A monadic action returning the shuffled (or not) train data
    let iterTrain = (if shuffleTrain then shuffleM else pure) train''

    -- | A monadic action returning a list of minibatches
    let iterBatches = case batchSize of
            Just n -> chunksOf n <$> iterTrain
            Nothing -> pure [train']

    -- | A list of successive batches over all iterations
    b <- concat <$> replicateM numIter iterBatches

    let m = snd $ last $ optimizer b cost model0
    pure (evalCost cost holdout m, (`runModel` m) <$> test')
  where
    optimizer = case optimizerChoice of
        GradientDescent -> gd learningRate learningRateDecay
        AdaGrad -> adagrad learningRate learningRateDecay 1e-6
        RMSProp -> rmsprop learningRate learningRateDecay 0.9 1e-6

data MeanVar a = MeanVar
    { mvSum :: !a
    , mvSumSquared :: !a
    , mvCount :: !Int
    }

instance (Additive m, Num a) => Monoid (MeanVar (m a)) where
    mempty = MeanVar zero zero 0
    (MeanVar x y z) `mappend` (MeanVar x' y' z') = MeanVar (x ^+^ x') (y ^+^ y') (z + z')

meanVar :: (Additive m, Num a) => m a -> MeanVar (m a)
meanVar x = MeanVar x (liftI2 (*) x x) 1

standardize :: (Foldable f, Functor f, Additive m, Floating a, Ord a)
    => f (m a) -> f (m a, b) -> (f (m a), f (m a, b))
standardize test train = (corr <$> test, first corr <$> train)
  where
    MeanVar s ss (fromIntegral -> n) = foldMap (meanVar . fst) train
    means = (/n) <$> s -- s / n = mean
    stdevs = sqrt <$> ((/n) <$> ss ^-^ liftI2 (*) means means) -- sqrt(E(X^2) - (EX^2)) = stddev
    stdevs' = (\x -> if abs x < 1e-6 then 1 else x) <$> stdevs -- if stddev = 0 then 1 else stddev
    corr x = liftI2 (/) (x ^-^ means) stdevs' -- (x - mean) / stddev
