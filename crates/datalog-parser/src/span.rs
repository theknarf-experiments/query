use crate::SrcId;
use std::{fmt, ops::Range};

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct Span {
    src: SrcId,
    range: (usize, usize),
}

impl Span {
    pub fn new(src: SrcId, range: Range<usize>) -> Self {
        assert!(range.start <= range.end);
        Self {
            src,
            range: (range.start, range.end),
        }
    }

    pub fn src(&self) -> SrcId {
        self.src
    }

    pub fn start(&self) -> usize {
        self.range.0
    }

    pub fn end(&self) -> usize {
        self.range.1
    }

    pub fn range(&self) -> Range<usize> {
        self.start()..self.end()
    }

    pub fn union(self, other: Self) -> Self {
        assert_eq!(self.src, other.src, "span source ids must match");
        Self {
            src: self.src,
            range: (self.start().min(other.start()), self.end().max(other.end())),
        }
    }
}

impl fmt::Debug for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}:{:?}", self.src, self.range())
    }
}

impl chumsky::Span for Span {
    type Context = SrcId;
    type Offset = usize;

    fn new(src: SrcId, range: Range<usize>) -> Self {
        Span::new(src, range)
    }

    fn context(&self) -> Self::Context {
        self.src
    }

    fn start(&self) -> Self::Offset {
        self.range.0
    }

    fn end(&self) -> Self::Offset {
        self.range.1
    }
}

impl ariadne::Span for Span {
    type SourceId = SrcId;

    fn source(&self) -> &Self::SourceId {
        &self.src
    }

    fn start(&self) -> usize {
        self.range.0
    }

    fn end(&self) -> usize {
        self.range.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_creation() {
        let src = SrcId::repl();
        let span = Span::new(src, 0..10);

        assert_eq!(span.start(), 0);
        assert_eq!(span.end(), 10);
        assert_eq!(span.src(), src);
    }

    #[test]
    fn test_span_union() {
        let src = SrcId::repl();
        let span1 = Span::new(src, 5..10);
        let span2 = Span::new(src, 0..7);

        let union = span1.union(span2);
        assert_eq!(union.start(), 0);
        assert_eq!(union.end(), 10);
    }

    #[test]
    fn test_span_range() {
        let src = SrcId::repl();
        let span = Span::new(src, 5..15);

        assert_eq!(span.range(), 5..15);
    }
}
