{-# LANGUAGE RecordWildCards #-}
module Main where

import Control.Arrow
import Data.List.Split
import Data.Hashable
import qualified Engine as E
import Options.Applicative

data Options = Options
    { trainFile :: String
    , testFile :: String
    , outFile :: String
    , batchSize :: Maybe Int
    , numIter :: Int
    , shuffleTrain :: Bool
    , holdoutSize :: Int
    , l2Coef :: Double
    , learningRate :: Double
    , learningRateDecay :: Double
    , loss :: String
    , standardizeInput :: Bool
    , useAdagrad :: Bool
    , useRmsprop :: Bool
    , hashFeaturesMod :: Maybe Int
    }

parser :: Parser Options
parser = Options
    <$> strOption
        (long "train_file"
         <> help "Train data file")
    <*> strOption
        (long "test_file"
         <> help "Test data file")
    <*> strOption
        (short 'o'
         <> help "Output file")
    <*> optional (option auto
        (long "batch_size"
         <> help "Batch size"))
    <*> option auto
        (long "n_iter"
         <> help "Number of iterations")
    <*> switch
        (long "shuffle"
         <> help "Shuffle the dataset in each iteration")
    <*> option auto
        (long "holdout_size")
    <*> option auto
        (long "l2"
         <> help "reguularization coefficient")
    <*> option auto
        (short 'l'
         <> help "Learning rate")
    <*> option auto
        (long "decay_learning_rate"
         <> help "Learning rate decay")
    <*> strOption
        (long "loss"
         <> help "loss: squared | logistic"
         <> value "squared")
    <*> switch
        (long "standardize")
    <*> switch
        (long "adagrad")
    <*> switch
        (long "rmsprop")
    <*> optional (option auto
        (long "hash"))
        
run :: Options -> IO ()
run Options{..} = do
    testData <- map (>>= parseFeature) . parse <$> readFile testFile
    trainData <- map ((>>= parseFeature) . init &&& read . last) . parse <$> readFile trainFile
    let useLogistic = case loss of
             "logistic" -> True
             "squared" -> False
             _ -> error $ "Illegal loss: " ++ loss
    let optimizerChoice = case (useAdagrad, useRmsprop) of
            (True, False) -> E.AdaGrad
            (False, True) -> E.RMSProp
            (False, False) -> E.GradientDescent
            _ -> error "Contradictory settings: adagrad & rmsprop"
    (holdout_loss, res) <- E.run E.Settings{..}
    putStrLn $ "Holdout set loss: " ++ show holdout_loss
    writeFile outFile $ unlines $ map show res
  where
    parse = map (splitOn ",") . tail . lines

    parseFeature feat = case hashFeaturesMod of
        Nothing -> [read feat]
        Just m -> let h = hash feat `mod` m in
            replicate h 0 ++ [1] ++ replicate (m - h - 1) 0

main = run =<< execParser (info (helper <*> parser)
        (fullDesc
        <> progDesc "Linear/Logistic regression"))
