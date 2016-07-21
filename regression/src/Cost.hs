{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE TypeFamilies #-}
module Cost
    ( Cost(..)
    , l2
    , mean
    , logistic
    , square
    ) where

import Data.Monoid
import Model
import Models
import Linear

-- | A cost function. It takes the model, a batch of (input, output)
-- and returns a scalar cost.
newtype Cost m = Cost
    { getCost :: forall f a. (Foldable f, Functor f, Floating a) =>
        m a -> f (Output m a, Output m a) -> a }

liftCost1 :: (forall a. Floating a => a -> a) -> Cost m -> Cost m
liftCost1 f (Cost c) = Cost $ \m b -> f (c m b)

liftCost2 :: (forall a. Floating a => a -> a -> a) -> Cost m -> Cost m -> Cost m
liftCost2 f (Cost c) (Cost c') = Cost $ \m b -> f (c m b) (c' m b)

instance Num (Cost m) where
    (+) = liftCost2 (+)
    (-) = liftCost2 (-)
    (*) = liftCost2 (*)
    negate = liftCost1 negate
    abs = liftCost1 abs
    signum = liftCost1 signum
    fromInteger x = Cost $ \_ _ -> fromInteger x

instance Fractional (Cost m) where
    (/) = liftCost2 (/)
    recip = liftCost1 recip
    fromRational x = Cost $ \_ _ -> fromRational x

instance Floating (Cost m) where
    pi = Cost $ \_ _ -> pi
    exp = liftCost1 exp
    log = liftCost1 log
    sqrt = liftCost1 sqrt
    (**) = liftCost2 (**)
    logBase = liftCost2 logBase
    sin = liftCost1 sin
    cos = liftCost1 cos
    tan = liftCost1 tan
    asin = liftCost1 asin
    acos = liftCost1 acos
    atan = liftCost1 atan
    sinh = liftCost1 sinh
    cosh = liftCost1 cosh
    tanh = liftCost1 tanh
    asinh = liftCost1 asinh
    acosh = liftCost1 acosh
    atanh = liftCost1 atanh

-- | L2 norm (excluding biases)
l2 :: HasL2 m => Cost m
l2 = Cost $ const . getL2

data Mean a = Mean !a !Int

instance Num a => Monoid (Mean a) where
    mempty = Mean 0 0
    Mean a x `mappend` Mean b y = Mean (a + b) (x + y)

toMean :: a -> Mean a
toMean = flip Mean 1

getMean :: Fractional a => Mean a -> a
getMean (Mean x n) = x / fromIntegral n

-- | An auxiliary function averaging some statictic over the minibatch,
-- e.g. 'mean' 'logistic' computes the mean logistic loss.
mean :: (Additive (Output m), Foldable (Output m)) =>
    (forall a. Floating a => a -> a -> a)
    -> Cost m
mean f = Cost $ \_ batch -> getMean (foldMap (foldMap toMean . uncurry (liftI2 f)) batch)

logistic :: Floating a => a -> a -> a
logistic predicted actual = -(actual * log (eps + predicted) + (1 - actual) * log (1 - predicted - eps))
  where
    eps = 1e-7 -- try to avoid NaNs

square :: Floating a => a -> a -> a
square x y = (x - y) * (x - y)
