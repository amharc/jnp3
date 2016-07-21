{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE DeriveFoldable #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MagicHash #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE UndecidableInstances #-}
module Models where

import Control.Applicative
import Data.Functor.Identity
import Data.Functor.Compose
import Data.Functor.Product
import Data.Functor.Classes
import Linear
import Model
import TH
import GHC.Prim
import GHC.TypeLits

-- | Boilerplate definitions
instance {-# OVERLAPPABLE #-} Metric m => HasL2 m where
    getL2 = quadrance 

instance (Applicative f, Additive g) => Additive (Compose f g) where
    zero = Compose $ pure zero
    liftU2 f (Compose x) (Compose y) = Compose $ liftA2 (liftU2 f) x y
    liftI2 f (Compose x) (Compose y) = Compose $ liftA2 (liftI2 f) x y

instance (Metric g, Applicative f, Foldable f) => Metric (Compose f g) where
    dot (Compose f) (Compose g) = sum $ dot <$> f <*> g

instance (HasL2 g, Functor f, Foldable f) => HasL2 (Compose f g) where
    getL2 (Compose f) = sum $ getL2 <$> f

instance (Additive f, Additive g) => Additive (Product f g) where
    zero = Pair zero zero
    liftU2 h (Pair x y) (Pair x' y') = Pair (liftU2 h x x') (liftU2 h y y')
    liftI2 h (Pair x y) (Pair x' y') = Pair (liftI2 h x x') (liftI2 h y y')

instance (Metric f, Metric g) => Metric (Product f g) where
    dot (Pair x y) (Pair x' y') = dot x x' + dot y y'

instance (HasL2 f, HasL2 g) => HasL2 (Product f g) where
    getL2 (Pair x y) = getL2 x + getL2 y

-- | Linear maps.
newtype LinearMap f g a = LinearMap' (Compose g f a)
    deriving (Show, Functor, Foldable, Traversable, Applicative, Additive, Metric, Show1, HasL2)

pattern LinearMap m = LinearMap' (Compose m)

instance (Metric f, Functor g) => Model (LinearMap f g) where
    type Input (LinearMap f g) = f
    type Output (LinearMap f g) = g
    runModel x (LinearMap m) = dot x <$> m

-- | Biases.
newtype Biased f a = Biased' (Product f Identity a)
    deriving (Show, Functor, Foldable, Traversable, Applicative, Additive, Show1, Metric)

pattern Biased f g = Biased' (Pair f g)

instance HasL2 f => HasL2 (Biased f) where
    getL2 (Biased f _) = getL2 f

-- | Affine maps.
newtype AffineMap f g a = AffineMap' (Compose g (Biased f) a)
    deriving (Show, Functor, Foldable, Traversable, Applicative, Additive, Metric, Show1, HasL2)

pattern AffineMap m = AffineMap' (Compose m)

instance (Metric f, Functor g) => Model (AffineMap f g) where
    type Input (AffineMap f g) = f
    type Output (AffineMap f g) = g
    runModel x (AffineMap m) = dot (Biased x (Identity 1)) <$> m

-- | Sequential composition of models.
newtype (f :-> g) a = Then' (Product f g a)
    deriving (Show, Functor, Foldable, Traversable, Applicative, Additive, Metric, Show1, HasL2)

pattern Then f g = Then' (Pair f g)

instance (Model f, Model g, Output f ~ Input g) => Model (f :-> g) where
    type Input (f :-> g) = Input f
    type Output (f :-> g) = Output g
    runModel x (Then f g) = runModel (runModel x f) g

-- | Some useful scalar functions
makeScalarFun [| \x -> 1/(1 + exp(-x)) |] "logistic"
makeScalarFun [| \x -> (x + abs x) / 2 |] "relu"
makeScalarFun [| tanh |] "tanh"

-- | Computes a scalar function over all elements of a Functor
newtype Over f (name :: Symbol) a = Over ()
    deriving (Show, Functor, Foldable, Traversable)

instance Applicative (Over f name) where
    pure _ = Over ()
    _ <*> _ = Over ()

instance Additive (Over f name) where
    zero = Over ()

instance Metric (Over f name) where
    dot _ _ = 0

instance KnownSymbol name => Show1 (Over f name) where
    showsPrec1 _ _ = (symbolVal' (proxy# :: Proxy# name) ++)

instance (Functor f, IsScalarFun name) => Model (Over f name) where
    type Input (Over f name) = f
    type Output (Over f name) = f
    runModel x _ = runScalarFun (proxy# :: Proxy# name) <$> x
