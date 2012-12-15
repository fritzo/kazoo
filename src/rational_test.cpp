
#include "rational.h"

void test_gcd ()
{
  using namespace Rational;

  LOG("Testing gcd");

  ASSERT_EQ(gcd(1,1), 1);
  ASSERT_EQ(gcd(1,2), 1);
  ASSERT_EQ(gcd(1,3), 1);
  ASSERT_EQ(gcd(1,4), 1);
  ASSERT_EQ(gcd(1,5), 1);
  ASSERT_EQ(gcd(1,6), 1);

  ASSERT_EQ(gcd(2,1), 1);
  ASSERT_EQ(gcd(2,2), 2);
  ASSERT_EQ(gcd(2,3), 1);
  ASSERT_EQ(gcd(2,4), 2);
  ASSERT_EQ(gcd(2,5), 1);
  ASSERT_EQ(gcd(2,6), 2);

  ASSERT_EQ(gcd(3,1), 1);
  ASSERT_EQ(gcd(3,2), 1);
  ASSERT_EQ(gcd(3,3), 3);
  ASSERT_EQ(gcd(3,4), 1);
  ASSERT_EQ(gcd(3,5), 1);
  ASSERT_EQ(gcd(3,6), 3);

  ASSERT_EQ(gcd(4,1), 1);
  ASSERT_EQ(gcd(4,2), 2);
  ASSERT_EQ(gcd(4,3), 1);
  ASSERT_EQ(gcd(4,4), 4);
  ASSERT_EQ(gcd(4,5), 1);
  ASSERT_EQ(gcd(4,6), 2);

  ASSERT_EQ(gcd(5,1), 1);
  ASSERT_EQ(gcd(5,2), 1);
  ASSERT_EQ(gcd(5,3), 1);
  ASSERT_EQ(gcd(5,4), 1);
  ASSERT_EQ(gcd(5,5), 5);
  ASSERT_EQ(gcd(5,6), 1);

  ASSERT_EQ(gcd(6,1), 1);
  ASSERT_EQ(gcd(6,2), 2);
  ASSERT_EQ(gcd(6,3), 3);
  ASSERT_EQ(gcd(6,4), 2);
  ASSERT_EQ(gcd(6,5), 1);
  ASSERT_EQ(gcd(6,6), 6);
}

void test_ball ()
{
  LOG("Testing ball");
  auto ball = Rational::ball_of_radius(4);
  ASSERT_EQ(ball.size(), 7);
  ASSERT_EQ(ball[0], Rational::Number(1, 3));
  ASSERT_EQ(ball[1], Rational::Number(1, 2));
  ASSERT_EQ(ball[2], Rational::Number(2, 3));
  ASSERT_EQ(ball[3], Rational::Number(1, 1));
  ASSERT_EQ(ball[4], Rational::Number(3, 2));
  ASSERT_EQ(ball[5], Rational::Number(2, 1));
  ASSERT_EQ(ball[6], Rational::Number(3, 1));
}

int main ()
{
  test_gcd();
  test_ball();

  return 0;
}

