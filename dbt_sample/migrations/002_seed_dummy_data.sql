USE dbt_sample;

INSERT INTO customers (customer_id, first_name, last_name, email, city, created_at) VALUES
(1, 'Alice',   'Nguyen',   'alice.nguyen@example.com',   'Austin',    '2024-01-05 10:00:00'),
(2, 'Brian',   'Cole',     'brian.cole@example.com',     'Seattle',   '2024-01-12 11:30:00'),
(3, 'Carla',   'Mendez',   'carla.mendez@example.com',   'Chicago',   '2024-02-01 09:15:00'),
(4, 'David',   'Patel',    'david.patel@example.com',    'Denver',    '2024-02-18 16:45:00'),
(5, 'Elena',   'Brooks',   'elena.brooks@example.com',   'Boston',    '2024-03-03 08:20:00');

INSERT INTO products (product_id, product_name, category, unit_price, created_at) VALUES
(101, 'Wireless Mouse',      'Electronics', 29.99,  '2024-01-01 00:00:00'),
(102, 'Mechanical Keyboard', 'Electronics', 89.50,  '2024-01-01 00:00:00'),
(103, 'Desk Lamp',           'Home',        24.00,  '2024-01-01 00:00:00'),
(104, 'Notebook Set',        'Office',      12.50,  '2024-01-01 00:00:00'),
(105, 'Coffee Mug',          'Home',         9.99,  '2024-01-01 00:00:00'),
(106, 'USB-C Hub',           'Electronics', 45.00,  '2024-01-01 00:00:00'),
(107, 'Monitor Stand',       'Electronics', 35.75,  '2024-01-01 00:00:00'),
(108, 'Ergonomic Chair',     'Furniture',  249.00,  '2024-01-01 00:00:00');

INSERT INTO orders (
    order_id, customer_id, product_id, quantity, unit_price, order_date, status, updated_at
) VALUES
(1001, 1, 102, 1, 89.50, '2024-03-10', 'completed', '2024-03-10 12:00:00'),
(1002, 1, 104, 3, 12.50, '2024-03-15', 'completed', '2024-03-15 12:00:00'),
(1003, 2, 101, 2, 29.99, '2024-03-12', 'completed', '2024-03-12 12:00:00'),
(1004, 2, 106, 1, 45.00, '2024-04-01', 'completed', '2024-04-01 12:00:00'),
(1005, 3, 108, 1, 249.00, '2024-04-05', 'completed', '2024-04-05 12:00:00'),
(1006, 3, 105, 4,  9.99, '2024-04-08', 'completed', '2024-04-08 12:00:00'),
(1007, 4, 107, 1, 35.75, '2024-04-11', 'shipped',    '2024-04-11 12:00:00'),
(1008, 4, 103, 2, 24.00, '2024-04-14', 'shipped',    '2024-04-14 12:00:00'),
(1009, 5, 101, 1, 29.99, '2024-04-20', 'completed',  '2024-04-20 12:00:00'),
(1010, 5, 102, 1, 89.50, '2024-04-22', 'returned',   '2024-04-22 12:00:00'),
(1011, 1, 106, 1, 45.00, '2024-05-02', 'completed',  '2024-05-02 12:00:00'),
(1012, 2, 104, 5, 12.50, '2024-05-06', 'completed',  '2024-05-06 12:00:00'),
(1013, 3, 107, 2, 35.75, '2024-05-09', 'processing', '2024-05-09 12:00:00'),
(1014, 4, 101, 1, 29.99, '2024-05-11', 'completed',  '2024-05-11 12:00:00'),
(1015, 5, 103, 1, 24.00, '2024-05-15', 'completed',  '2024-05-15 12:00:00');