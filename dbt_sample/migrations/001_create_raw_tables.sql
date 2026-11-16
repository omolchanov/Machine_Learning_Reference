-- Raw source tables for the dbt_sample project (MySQL).
-- Run against local Docker MySQL before dbt models.

CREATE DATABASE IF NOT EXISTS dbt_sample
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_unicode_ci;

USE dbt_sample;

DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id   INT            NOT NULL,
    first_name    VARCHAR(64)    NOT NULL,
    last_name     VARCHAR(64)    NOT NULL,
    email         VARCHAR(128)   NOT NULL,
    city          VARCHAR(64)    NOT NULL,
    created_at    DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id),
    UNIQUE KEY uq_customers_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE products (
    product_id    INT            NOT NULL,
    product_name  VARCHAR(128)   NOT NULL,
    category      VARCHAR(64)    NOT NULL,
    unit_price    DECIMAL(10, 2) NOT NULL,
    created_at    DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE orders (
    order_id      INT            NOT NULL,
    customer_id   INT            NOT NULL,
    product_id    INT            NOT NULL,
    quantity      INT            NOT NULL CHECK (quantity > 0),
    unit_price    DECIMAL(10, 2) NOT NULL,
    order_date    DATE           NOT NULL,
    status        VARCHAR(32)    NOT NULL,
    updated_at    DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (order_id),
    KEY idx_orders_customer_id (customer_id),
    KEY idx_orders_product_id (product_id),
    KEY idx_orders_order_date (order_date),
    KEY idx_orders_updated_at (updated_at),
    CONSTRAINT fk_orders_customer
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id),
    CONSTRAINT fk_orders_product
        FOREIGN KEY (product_id) REFERENCES products (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;