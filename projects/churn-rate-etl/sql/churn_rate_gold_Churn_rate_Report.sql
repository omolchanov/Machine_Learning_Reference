-- MySQL dump 10.13  Distrib 8.0.42, for Win64 (x86_64)
--
-- Host: mysql-25effa04-oleksandr-45fd.d.aivencloud.com    Database: churn_rate_gold
-- ------------------------------------------------------
-- Server version	8.0.35

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
SET @MYSQLDUMP_TEMP_LOG_BIN = @@SESSION.SQL_LOG_BIN;
SET @@SESSION.SQL_LOG_BIN= 0;

--
-- GTID state at the beginning of the backup 
--

SET @@GLOBAL.GTID_PURGED=/*!80000 '+'*/ '41d79771-1ea0-11f0-826c-da04134c830c:1-130,
b269f3cb-2128-11f0-8ddd-862ccfb0673b:1-24668';

--
-- Table structure for table `Churn_rate_Report`
--

DROP TABLE IF EXISTS `Churn_rate_Report`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Churn_rate_Report` (
  `id` int NOT NULL AUTO_INCREMENT,
  `state` varchar(2) DEFAULT NULL,
  `account_length` int DEFAULT NULL,
  `phone_number` varchar(8) DEFAULT NULL,
  `international_plan` varchar(3) DEFAULT NULL,
  `voice_mail_plan` varchar(3) DEFAULT NULL,
  `number_vmail_messages` int DEFAULT NULL,
  `total_day_minutes` decimal(4,1) DEFAULT NULL,
  `total_day_calls` int DEFAULT NULL,
  `total_day_charge` decimal(4,2) DEFAULT NULL,
  `total_eve_minutes` decimal(4,1) DEFAULT NULL,
  `total_eve_calls` int DEFAULT NULL,
  `total_eve_charge` decimal(4,2) DEFAULT NULL,
  `total_night_minutes` decimal(4,1) DEFAULT NULL,
  `total_night_calls` int DEFAULT NULL,
  `total_night_charge` decimal(4,2) DEFAULT NULL,
  `total_intl_minutes` decimal(3,1) DEFAULT NULL,
  `total_intl_calls` int DEFAULT NULL,
  `total_intl_charge` decimal(3,2) DEFAULT NULL,
  `customer_service_calls` int DEFAULT NULL,
  `churn` varchar(5) DEFAULT NULL,
  `first_name` varchar(255) NOT NULL,
  `last_name` varchar(255) NOT NULL,
  `balance` int DEFAULT '0',
  `join_date` date NOT NULL,
  `churn_date` date DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `Churn_rate_Report`
--

LOCK TABLES `Churn_rate_Report` WRITE;
/*!40000 ALTER TABLE `Churn_rate_Report` DISABLE KEYS */;
/*!40000 ALTER TABLE `Churn_rate_Report` ENABLE KEYS */;
UNLOCK TABLES;
SET @@SESSION.SQL_LOG_BIN = @MYSQLDUMP_TEMP_LOG_BIN;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-05-08 22:59:34
