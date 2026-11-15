use futures_lite::{Stream, StreamExt, stream};
use std::io;

/// Decode byte-oriented stdin without making read boundaries UTF-8 boundaries.
pub(crate) fn text_stream<I: Stream<Item = io::Result<Vec<u8>>> + Unpin>(
    input: I,
) -> impl Stream<Item = io::Result<String>> {
    stream::unfold(
        (input, Vec::new(), false),
        |(mut input, mut pending, done)| async move {
            if done {
                return None;
            }
            loop {
                let bytes = match input.next().await {
                    Some(Ok(bytes)) => Some(bytes),
                    Some(Err(error)) => return Some((Err(error), (input, pending, true))),
                    None => None,
                };
                let ended = bytes.is_none();
                if let Some(bytes) = bytes {
                    pending.extend(bytes);
                }
                let (valid, invalid) = match std::str::from_utf8(&pending) {
                    Ok(_) => (pending.len(), false),
                    Err(error) => (error.valid_up_to(), error.error_len().is_some() || ended),
                };
                if invalid {
                    return Some((
                        Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "stdin contains invalid or incomplete UTF-8",
                        )),
                        (input, pending, true),
                    ));
                }
                if valid != 0 {
                    let text = String::from_utf8(pending.drain(..valid).collect())
                        .expect("validated UTF-8 prefix");
                    return Some((Ok(text), (input, pending, ended)));
                }
                if ended {
                    return None;
                }
            }
        },
    )
}

pub(crate) fn pipe_input(
    fd: std::os::fd::OwnedFd,
) -> io::Result<impl Stream<Item = io::Result<Vec<u8>>>> {
    let input = async_io::Async::new(fd)?;
    Ok(stream::unfold(Some(input), |input| async move {
        let input = input?;
        // Transport read size, independent of the library's text policy.
        let mut bytes = vec![0u8; 4096];
        match input
            .read_with(|fd| rustix::io::read(fd, &mut bytes).map_err(Into::into))
            .await
        {
            Ok(0) => None,
            Ok(count) => {
                bytes.truncate(count);
                Some((Ok(bytes), Some(input)))
            }
            Err(error) => Some((Err(error), None)),
        }
    }))
}

pub(crate) fn file_input(file: std::fs::File) -> impl Stream<Item = io::Result<Vec<u8>>> {
    stream::unfold(Some(file), |file| async move {
        let mut file = file?;
        match tokio::task::spawn_blocking(move || {
            use std::io::Read;
            let mut bytes = vec![0u8; 4096];
            let result = file.read(&mut bytes).map(|count| {
                bytes.truncate(count);
                bytes
            });
            (file, result)
        })
        .await
        {
            Ok((_, Ok(bytes))) if bytes.is_empty() => None,
            Ok((file, Ok(bytes))) => Some((Ok(bytes), Some(file))),
            Ok((_, Err(error))) => Some((Err(error), None)),
            Err(error) => Some((Err(io::Error::other(error)), None)),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_lite::{StreamExt, future};
    #[test]
    fn split_utf8_and_invalid_eof() {
        future::block_on(async {
            let text = "今天阳光很好。こんにちは。안녕하세요. Hello!";
            let mut stream = Box::pin(text_stream(stream::iter(text.bytes().map(|b| Ok(vec![b])))));
            let mut actual = String::new();
            while let Some(part) = stream.next().await {
                actual.push_str(&part.unwrap());
            }
            assert_eq!(actual, text);
            for invalid in [&b"\xff"[..], &b"\xe6\xb3"[..]] {
                let mut stream = Box::pin(text_stream(stream::iter(
                    invalid.iter().map(|&b| Ok(vec![b])),
                )));
                assert_eq!(
                    stream.next().await.unwrap().unwrap_err().kind(),
                    io::ErrorKind::InvalidData
                );
                assert!(stream.next().await.is_none());
            }
        });
    }
}
