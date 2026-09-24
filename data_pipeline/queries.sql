-- 1. SELECT / WHERE
SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4
ORDER BY rating DESC, price_gbp DESC;

-- 2. ORDER BY
SELECT title, price_gbp, price_inr
FROM books
ORDER BY price_inr DESC;

-- 3. LIMIT
SELECT title, rating, price_gbp
FROM books
ORDER BY rating DESC, price_gbp DESC
LIMIT 10;

-- 4. DISTINCT
SELECT DISTINCT category_name
FROM categories
ORDER BY category_name;

-- 5. BETWEEN
SELECT title, price_gbp, rating
FROM books
WHERE price_gbp BETWEEN 10 AND 30
ORDER BY price_gbp;

-- 6. JOIN
SELECT c.category_name, b.title, b.rating, b.price_gbp, b.price_inr
FROM books b
JOIN categories c ON b.category_id = c.category_id
ORDER BY c.category_name, b.rating DESC, b.price_gbp DESC
LIMIT 10;
