{-# LANGUAGE Rank2Types #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}
module Optimizer where

import Data.Bifunctor
import Numeric.AD
import Model
import Linear
import Cost

type Batch f m a = f (Input m a, Output m a)

type Optimizer a = forall m f.
       (Model m, Functor m, Foldable m, Traversable m, Additive m, Foldable f, Functor f)
    => [Batch f m a] -- ^ List of mini-batches
    -> Cost m -- ^ The cost function
    -> m a -- ^ The initial model
    -> [(a, m a)] -- ^ List of successive models (together with their costs)

gd :: Floating a => a -> a -> Optimizer a
gd lr0 lr_decay batches cost m0 = fmap (\(x, _, y) -> (x, y)) $ tail $
    scanl go (undefined, lr0, m0) batches
  where
    go (_, !lr, !m) b = (c, lr * lr_decay, m')
      where
        (c, g) = evalCost' cost b m
        m' = m ^-^ fmap (lr *) g

adagrad :: Floating a => a -> a -> a -> Optimizer a
adagrad lr0 lr_decay eps batches cost m0 = fmap (\(x, y, _, _) -> (x, y)) $ tail $
    scanl go (undefined, m0, lr0, zero) batches
  where
    -- (cost, model, learning rate, accumulated gradient)
    go (_, !m, !lr, !acc) b = (c, m', lr * lr_decay, acc')
      where
        (c, g) = evalCost' cost b m
        acc' = acc ^+^ liftI2 (*) g g
        g' = liftI2 (/) g $ (\x -> sqrt (x + eps)) <$> acc'
        m' = m ^-^ fmap (lr *) g'

rmsprop :: Floating a => a -> a -> a -> a -> Optimizer a
rmsprop lr0 lr_decay rho eps batches cost m0 = fmap (\(x, y, _, _) -> (x, y)) $ tail $
    scanl go (undefined, m0, lr0, zero) batches
  where
    -- (cost, model, learning rate, accumulated gradient)
    go (_, !m, !lr, !acc) b = (c, m', lr * lr_decay, acc')
      where
        (c, g) = evalCost' cost b m
        acc' = fmap (rho *) acc ^+^ fmap ((1 - rho) *) (liftI2 (*) g g)
        g' = liftI2 (/) g $ (\x -> sqrt (x + eps)) <$> acc'
        m' = m ^-^ fmap (lr *) g'

-- | Evaluates the cost function for a model over a batch.
evalCost :: (Floating a, Model m, Foldable f, Functor f)
            => Cost m -> Batch f m a -> m a -> a
evalCost (Cost c) b m = c m $ (\(i, o) -> (runModel i m, o)) <$> b

-- | Evaluates the cost function and finds its gradient.
evalCost' :: (Floating a, Model m, Traversable m, Foldable f, Functor f)
            => Cost m -> Batch f m a -> m a -> (a, m a)
evalCost' c b = grad' (evalCost c (bimap (auto <$>) (auto <$>) <$> b))
