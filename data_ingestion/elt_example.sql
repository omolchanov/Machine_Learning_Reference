DELIMITER $$

DROP PROCEDURE IF EXISTS clean_and_transform_sales $$
CREATE PROCEDURE clean_and_transform_sales()
BEGIN
    DECLARE row_count INT DEFAULT 0;
    DECLARE q1_index INT DEFAULT 0;
    DECLARE q3_index INT DEFAULT 0;
    DECLARE q1 DECIMAL(10,2) DEFAULT 0;
    DECLARE q3 DECIMAL(10,2) DEFAULT 0;
    DECLARE iqr DECIMAL(10,2) DEFAULT 0;
    DECLARE lower_bound DECIMAL(10,2) DEFAULT 0;
    DECLARE upper_bound DECIMAL(10,2) DEFAULT 0;

    proc_block: BEGIN  -- <== Labeled block so LEAVE can work

        -- Count non-null amount rows to calculate percentiles
        SELECT COUNT(*) INTO row_count
        FROM staging_sales
        WHERE amount IS NOT NULL;

        IF row_count = 0 THEN
            INSERT INTO log_table(message)
            VALUES ('No valid rows in staging_sales for IQR calculation');
            LEAVE proc_block;  -- <== Exit the labeled block safely
        END IF;

        SET q1_index = FLOOR(0.25 * row_count);
        SET q3_index = FLOOR(0.75 * row_count);

        -- Calculate Q1
        SELECT amount INTO q1
        FROM (
            SELECT amount
            FROM staging_sales
            WHERE amount IS NOT NULL
            ORDER BY amount
            LIMIT 1 OFFSET q1_index
        ) AS q1_sub;

        -- Calculate Q3
        SELECT amount INTO q3
        FROM (
            SELECT amount
            FROM staging_sales
            WHERE amount IS NOT NULL
            ORDER BY amount
            LIMIT 1 OFFSET q3_index
        ) AS q3_sub;

        SET iqr = q3 - q1;
        SET lower_bound = q1 - 1.5 * iqr;
        SET upper_bound = q3 + 1.5 * iqr;

        INSERT INTO log_table(message)
        VALUES (
            CONCAT('IQR computed: Q1=', q1, ', Q3=', q3, ', IQR=', iqr,
                   ', lower_bound=', lower_bound, ', upper_bound=', upper_bound)
        );

        -- Insert clean rows only, filtering nulls and capping outliers
        INSERT INTO sales (sale_id, customer_name, amount, purchase_date)
        SELECT
            id,
            customer_name,
            CASE
                WHEN amount < lower_bound THEN lower_bound
                WHEN amount > upper_bound THEN upper_bound
                ELSE amount
            END AS capped_amount,
            purchase_date
        FROM staging_sales
        WHERE
            customer_name IS NOT NULL AND
            amount IS NOT NULL AND
            purchase_date IS NOT NULL;

        INSERT INTO log_table(message)
        VALUES ('Cleaned data inserted into sales from staging_sales (no modification to staging)');

    END proc_block;
END $$
DELIMITER ;

SET SQL_SAFE_UPDATES = 0;
CALL clean_and_transform_sales();
SET SQL_SAFE_UPDATES = 1;  -- (optional) re-enable
